import json
import os
import time
import random
import math
import datetime
import re
from SPARQLWrapper import SPARQLWrapper, JSON

# import openai
from collections import Counter
import tiktoken
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import unicodedata
from openai import OpenAI
import math

KG_QUERY_COUNT_SOG = 0


def execute_sparql_query(query, sparql_endpoint="http://localhost:8890/sparql"):
    """
    Execute a SPARQL query and return the results (list of bindings).
    """
    global KG_QUERY_COUNT_SOG
    KG_QUERY_COUNT_SOG += 1

    max_retries = 3
    delay_seconds = 5

    sparql = SPARQLWrapper(sparql_endpoint)
    sparql.setReturnFormat(JSON)
    sparql.setQuery(query)
    for attempt in range(max_retries):
        try:
            results = sparql.query().convert()
            # If it's an ASK query, return the boolean
            if "ask" in query.lower():
                return results["boolean"]
            else:
                return results["results"]["bindings"]
        except Exception as e:
            print(f"[Attempt {attempt+1}/{max_retries}] SPARQL query error: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay_seconds} seconds...")
                time.sleep(delay_seconds)
            else:
                print("Max retries reached. Returning an empty result.")
                return []


def flatten_sparql_bindings(data: dict) -> dict:
    """
    For each row in data['results']['bindings'], flatten the nested dict structure
    so that we only keep the 'value' string for the keys of interest:
    property, propertyLabel, value, valueLabel.

    Modifies 'data' in place and also returns it for convenience.
    """
    bindings = data
    for row in bindings:
        for var_name in ["property", "propertyLabel", "value", "valueLabel"]:
            if var_name in row and "value" in row[var_name]:
                row[var_name] = row[var_name]["value"]
    return data


def remove_irrelevant_info(sparql_result: dict) -> dict:
    """
    Given a SPARQL result of the form:
      {
        "head": {"vars": [...]},
        "results": {"bindings": [ { "property": "...", "value": "..." }, ... ]}
      }
    Remove any lines (bindings) that appear irrelevant for normal QA usage,
    such as repeated foreign-language labels, property keys, Wikipedia keys, etc.

    Returns the same structure but with fewer 'bindings'.
    """

    # 1) We'll define sets or substrings of properties we consider "irrelevant"
    #    This can be refined to your exact needs:
    IRRELEVANT_PROPERTY_SUBSTRS = [
        "/key/",  # e.g. 'type.object.key' or any '.../key/wikipedia...'
        "dataworld.freeq",  # internal data references
        "wikipedia",  # all Wikipedia references
        "topic_equivalent_webpage",  # webpages
    ]
    IRRELEVANT_PROPERTIES_EXACT = {
        # "http://www.w3.org/2000/01/rdf-schema#label",   # typically repeated multi-language labels
        "http://rdf.freebase.com/ns/type.object.key",  # the actual property for "key"
        "http://rdf.freebase.com/ns/common.topic.description",
        "http://rdf.freebase.com/ns/common.topic.alias",
    }

    # 2) We'll retrieve the "bindings" list
    bindings = sparql_result.get("results", {}).get("bindings", [])
    filtered_bindings = []

    # 3) Iterate over each row
    for row in bindings:
        prop_val = row.get("property", "")
        val_val = row.get("value", "")
        val_label = row.get("valueLabel", "")

        # 1) Skip if property starts with "http"
        if prop_val.startswith("http://www.w3.org"):
            # print(f"prop_val: {prop_val}")
            continue

        # --- Criteria to skip:
        #  (a) If the property is exactly in IRRELEVANT_PROPERTIES_EXACT
        if prop_val in IRRELEVANT_PROPERTIES_EXACT:
            continue

        #  (b) If the property string has any IRRELEVANT_PROPERTY_SUBSTRS
        if any(substr in prop_val for substr in IRRELEVANT_PROPERTY_SUBSTRS):
            continue

        # If we pass all checks => keep
        filtered_bindings.append(row)

    # 4) Return the same shape, but with fewer lines
    return {
        "head": sparql_result.get("head", {}),
        "results": {"bindings": filtered_bindings},
    }


def remove_freebase_ns_prefix(sparql_result: dict) -> dict:
    """
    Remove "http://rdf.freebase.com/ns/" from the property and value fields.
    Returns a modified copy of sparql_result.
    """
    PREFIX = "http://rdf.freebase.com/ns/"

    # Copy the outer structure to avoid mutating original
    new_data = {
        "head": sparql_result.get("head", {}).copy(),
        "results": {"bindings": []},
    }

    bindings = sparql_result.get("results", {}).get("bindings", [])
    for row in bindings:
        # Copy row so we don't alter the original
        new_row = dict(row)

        for field in ["property", "propertyLabel", "value", "valueLabel"]:
            if field in new_row and isinstance(new_row[field], str):
                new_row[field] = new_row[field].replace(PREFIX, "")

        new_data["results"]["bindings"].append(new_row)

    return new_data


# Peter's clever idea
def get_adjacent_relations_and_entities_freebase(
    # question: str,
    entity: str,
    direction: str,
    sparql_endpoint: str = "http://localhost:8890/sparql",
    properties_to_filter_for: list = None,
    return_format="markdown-short",
) -> dict:
    """
    Build and execute a SPARQL query on a Freebase triple store that retrieves adjacent properties,
    property labels, values, and value labels in the specified direction for a given entity.

    Args:
        entity: The entity (e.g., 'ns:m.03_dwn') or a full URI
                (e.g. '<http://rdf.freebase.com/ns/m.03_dwn>')
                whose adjacent relations and entities we want to fetch.
        direction: The direction of the relation to consider: 'outgoing' or 'incoming'.
        sparql_endpoint: The SPARQL endpoint URL with Freebase data loaded (default local Virtuoso).
        properties_to_filter_for: Optional list of properties to filter for. If provided, only
                                 entities connected through these properties will be returned.

    Returns:
        A dictionary with "head" and "results" keys, similar to typical SPARQL JSON:
          {
            "head": {"vars": [...]},
            "results": {"bindings": [ ... ] }
          }
        where each binding row has:
          ?property, ?propertyLabel, ?value, ?valueLabel
    """

    # If the entity is written as 'ns:m.03_dwn' or '/m/03_dwn',
    # convert it to a full URI if needed:
    entity_uri = entity.strip()
    if not entity_uri.startswith("<"):
        # e.g. 'ns:m.03_dwn' => '<http://rdf.freebase.com/ns/m.03_dwn>'
        # or if user provided '/m/03_dwn', replace the slash:
        if entity_uri.startswith("ns:"):
            # 'ns:m.03_dwn' => 'm.03_dwn'
            entity_uri = entity_uri[len("ns:") :]
            entity_uri = entity_uri.replace("/", ".")
            entity_uri = f"<http://rdf.freebase.com/ns/{entity_uri}>"
        elif entity_uri.startswith("/"):
            # '/m/03_dwn' => 'm.03_dwn'
            entity_uri = entity_uri[1:].replace("/", ".")
            entity_uri = f"<http://rdf.freebase.com/ns/{entity_uri}>"
        else:
            # fallback: treat it as an ID
            entity_uri = entity_uri.replace("/", ".")
            entity_uri = f"<http://rdf.freebase.com/ns/{entity_uri}>"

    if direction == "outgoing":
        # entity is subject
        triple_pattern = f"{entity_uri} ?property ?value ."
    elif direction == "incoming":
        # entity is object
        triple_pattern = f"?value ?property {entity_uri} ."
    else:
        raise ValueError("direction must be 'outgoing' or 'incoming'.")

    # Create VALUES clause if properties_to_filter_for is provided
    values_clause = ""
    if properties_to_filter_for and len(properties_to_filter_for) > 0:
        property_uris = []
        for prop in properties_to_filter_for:
            # Properties come in format like 'location.location.contains'
            # Convert directly to URI format
            prop = prop.strip()
            if prop.startswith("<") and prop.endswith(">"):
                # Already a full URI
                prop_uri = prop
            else:
                # Standard dot-separated format - just wrap in URI
                prop_uri = f"<http://rdf.freebase.com/ns/{prop}>"
            property_uris.append(prop_uri)

        values_clause = f"VALUES ?property {{ {' '.join(property_uris)} }}"

    # We'll retrieve data in pages to avoid timeouts
    page_size = 10000  # Adjust as needed
    offset = 0
    all_bindings = []  # we accumulate rows from each "page"

    while True:

        # Construct the SPARQL query.
        query = f"""
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT DISTINCT ?property ?propertyLabel ?value ?valueLabel
        WHERE {{
        {values_clause}
        {triple_pattern}

        # Try to retrieve English labels for the property
        OPTIONAL {{
            ?property ns:type.object.name ?propertyLabel .
            FILTER (lang(?propertyLabel) = "en")
        }}

        # Try to retrieve English labels for the value
        OPTIONAL {{
            ?value ns:type.object.name ?valueLabel .
            FILTER (lang(?valueLabel) = "en")
        }}
        }}
        # ORDER BY ?propertyLabel ?valueLabel
        LIMIT {page_size}
        OFFSET {offset}
        """

        page_results = execute_sparql_query(query, sparql_endpoint)

        if not page_results:
            break

        page_bindings = page_results

        all_bindings.extend(page_bindings)

        if len(page_bindings) < page_size:
            break

        offset += page_size

    flattened_results = {
        "head": {"vars": ["property", "propertyLabel", "value", "valueLabel"]},
        "results": {"bindings": flatten_sparql_bindings(all_bindings)},
    }

    filtered_results = remove_irrelevant_info(flattened_results)
    # print(f"Results (filtered): \n{filtered_results}\n")

    filtered_results = remove_freebase_ns_prefix(filtered_results)
    # print(f"Results (removed freebase ns prefix): \n{filtered_results}\n")

    filtered_results = remove_duplicates_from_sparql_result(filtered_results)
    # print(f"Results (removed duplicates): \n{filtered_results}\n\n")

    # print(f"len(filtered_results['results']['bindings']): {len(filtered_results['results']['bindings'])}")

    # Check if we have too many results and properties_to_filter_for was not provided
    if (
        len(filtered_results["results"]["bindings"]) > 50
        and not properties_to_filter_for
        and return_format == "markdown-short"
    ):
        # Extract unique properties and their labels
        unique_properties = {}
        for triple in filtered_results["results"]["bindings"]:
            prop = triple.get("property", "")
            prop_label = triple.get("propertyLabel", "")
            if prop and prop not in unique_properties:
                unique_properties[prop] = prop_label

        # print(f"unique_properties: {unique_properties}")
        unique_properties_table = convert_properties_to_table(unique_properties)
        # print(f"unique_properties_table:\n\n{unique_properties_table}")

        return unique_properties_table

    top_k = 2000
    if len(filtered_results["results"]["bindings"]) >= top_k:
        # Slice to first top_k results to avoid overload
        filtered_results["results"]["bindings"] = filtered_results["results"][
            "bindings"
        ][:top_k]

    if "markdown" in return_format:
        return convert_tool_result_to_table(filtered_results)

    # print(f"filtered_results_table:\n\n{filtered_results_table}")
    elif "json" in return_format:
        return str(filtered_results)
        # filtered_results_table = json.dumps(filtered_results, indent=2)

    raise ValueError("return_format must be 'markdown-short' or 'markdown' or 'json'.")
    return filtered_results_table


def remove_duplicates_from_sparql_result(data):
    """
    Remove duplicate entries from SPARQL query results.

    Duplicates are identified by the combination of (property, propertyLabel, value, valueLabel).
    When duplicates exist, the first occurrence is kept.

    Args:
        data: Dictionary containing SPARQL query results with 'head' and 'results' keys

    Returns:
        Dictionary with duplicates removed
    """
    if (
        not isinstance(data, dict)
        or "results" not in data
        or "bindings" not in data["results"]
    ):
        return data

    # Create a new list for unique bindings
    unique_bindings = []
    seen_tuples = set()

    for binding in data["results"]["bindings"]:
        # Create a unique key from all four fields
        property_val = binding.get("property", "")
        property_label = binding.get("propertyLabel", "")
        value_val = binding.get("value", "")
        value_label = binding.get("valueLabel", "")

        # Use all four fields as the unique identifier
        unique_key = (property_val, property_label, value_val, value_label)

        # Only add if we haven't seen this exact combination before
        if unique_key not in seen_tuples:
            seen_tuples.add(unique_key)
            unique_bindings.append(binding)

    # Create the cleaned result
    cleaned_data = {
        "head": data.get("head", {}),
        "results": {"bindings": unique_bindings},
    }

    return cleaned_data


def convert_properties_to_table(unique_properties):
    """Convert unique_properties dictionary to compact table format"""
    # Define headers
    headers = ["property", "propertyLabel"]

    # Create table with headers and separator row
    table = [headers, ["--" for _ in headers]]

    # Add property rows
    for prop, label in unique_properties.items():
        table.append([prop, label])

    # Convert to pipe-separated string
    return "\n".join("|".join(row) for row in table)


def convert_tool_result_to_table(tool_content):
    """Convert JSON tool result to compact
    format"""
    tool_content = (
        json.loads(tool_content) if isinstance(tool_content, str) else tool_content
    )
    head = tool_content["head"]["vars"]
    # {"vars": ["property", "propertyLabel", "value", "valueLabel"]}
    table = [head, ["--" for _ in head]]
    for result in tool_content["results"]["bindings"]:
        table.append([result.get(col, "") for col in head])
    # convert to string
    return f"{len(table)-2} rows:\n" + "\n".join("|".join(row) for row in table)


def exact_match(pred, gold):
    pred_clean = pred.strip().lower().replace(" ", "")
    gold_clean = gold.strip().lower().replace(" ", "")
    return (
        pred_clean == gold_clean or pred_clean in gold_clean or gold_clean in pred_clean
    )


def tool_call_to_dict(tool_call):
    """
    Convert a ChatCompletionMessageToolCall object into a JSON-serializable dict.
    """
    return {
        "id": tool_call.id,
        "type": tool_call.type,
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,  # usually a JSON string
        },
    }


def extract_answers_inside_curly_braces(text):
    # Find all occurrences of { ... }
    matches = re.findall(r"\{([^}]*)\}", text)
    if matches:
        # Return a list of stripped matches
        return [m.strip() for m in matches]
    else:
        return []


def extract_answers_after_final_answers(text):
    """
    Returns all content after 'Final Answer:' (case-insensitive) as a list of words.
    Removes punctuation from each word.
    If not found, returns an empty list.
    """
    import string

    pattern = re.compile(r"(?i)final\s*answer:\s*(.*)", re.DOTALL)
    match = pattern.search(text)
    if match:
        content = match.group(1).strip()
        # Remove punctuation and split into words
        # This removes common punctuation but keeps content inside curly braces intact
        translator = str.maketrans(
            "", "", string.punctuation.replace("{", "").replace("}", "")
        )
        cleaned_content = content.translate(translator)
        # Split into words and remove empty strings
        words = [word.strip() for word in cleaned_content.split() if word.strip()]
        return words
    else:
        return []


def fill_empty_property_labels(
    all_bindings, sparql_endpoint="http://localhost:8890/sparql"
):
    """
    For each binding in all_bindings, if 'propertyLabel' is empty,
    parse out the property's ID (e.g., P8138) from the property URI,
    then query for its rdfs:label in English, and fill that in.
    """
    for row in all_bindings:
        # Ensure row has 'propertyLabel' -> 'value'
        if "propertyLabel" not in row:
            continue

        label_value = row["propertyLabel"].get("value", "")
        if label_value.strip() != "":
            # Already has a label, skip
            continue

        # 1) Extract the property ID from property URI (like "http://www.wikidata.org/prop/direct/P8138")
        if "property" not in row or "value" not in row["property"]:
            continue

        property_uri = row["property"]["value"]

        # Typically looks like: "http://www.wikidata.org/prop/direct/P8138"
        match = re.search(r"(P\d+)", property_uri)
        if not match:
            continue

        property_id = match.group(1)  # e.g. "P8138"

        # 2) Query to get the property label from wd:Pxxxx
        # We'll define a quick query for the property itself, not the wdt: direct property.
        # So "wd:P8138" is the item that should have the actual rdfs:label.
        query = f"""
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?label
        WHERE {{
          wd:{property_id} rdfs:label ?label .
          FILTER (LANG(?label) = "en")
        }}
        """

        # 3) Execute the SPARQL query
        results = execute_sparql_query(query, sparql_endpoint)
        if results:
            # results is a list of { 'label': { 'type':..., 'value':...} }
            label_str = results[0]["label"]["value"]
            row["propertyLabel"]["value"] = label_str
        else:
            # fallback: use property_id if no label found
            row["propertyLabel"]["value"] = property_id

    return all_bindings


def filter_non_english_rows(all_bindings):
    """
    Return only those rows in all_bindings whose valueLabel is English or lacks a language tag.
    """
    filtered = []
    for row in all_bindings:
        # Safely extract the 'valueLabel' dict
        val_label = row.get("value", {})

        # If there's no xml:lang key or it's "en", we keep the row
        lang = val_label.get("xml:lang", "")
        if lang == "en" or lang == "":
            filtered.append(row)
    return filtered


def shorten_property_and_value_uris_qald(all_bindings):
    """
    For each row in all_bindings, take the 'property' -> 'value' field
    (the full URI, like 'http://www.wikidata.org/prop/direct/P166')
    and reduce it to 'wdt:P166'.

    Similarly, take 'value' -> 'value' field
    (like 'http://www.wikidata.org/entity/Q64746327') and reduce it to 'wd:Q64746327'.
    """
    for row in all_bindings:
        # 1) Convert property URI from e.g. "http://www.wikidata.org/prop/direct/P166" -> "P166"
        if "property" in row and "value" in row["property"]:
            prop_uri = row["property"]["value"]
            match = re.search(
                r"^http://www\.wikidata\.org/prop/direct/(P\d+)$", prop_uri
            )
            if match:
                row["property"]["value"] = "wdt:" + match.group(1)  # e.g. "wdt:P166"

        # 2) Convert value URI from e.g. "http://www.wikidata.org/entity/Q213" -> "Q213"
        if "value" in row and "value" in row["value"]:
            val_uri = row["value"]["value"]
            match2 = re.search(r"^http://www\.wikidata\.org/entity/(Q\d+)$", val_uri)
            if match2:
                row["value"]["value"] = "wd:" + match2.group(1)  # e.g. "wd:Q213"
    return all_bindings


def keep_only_direct_prop_rows_qald(all_bindings):
    """
    Retain rows in all_bindings whose 'property' URI matches one of:
      - http://www.wikidata.org/prop/Pxxxx
      - http://www.wikidata.org/prop/direct/Pxxxx
      - http://www.wikidata.org/prop/statement/Pxxxx
      - http://www.wikidata.org/prop/qualifier/Pxxxx

    Do each match in a separate if block.
    """
    cleaned = []
    for row in all_bindings:
        if "property" not in row:
            continue
        prop_uri = row["property"].get("value", "")

        # Keep if it matches http://www.wikidata.org/prop/direct/P + digits
        if re.match(r"^http://www\.wikidata\.org/prop/direct/P\d+$", prop_uri):
            cleaned.append(row)

    return cleaned


def get_adjacent_relations_and_entities_wikidata(
    # question: str,
    entity: str,
    direction: str,
    sparql_endpoint: str = "http://localhost:8890/sparql",
    properties_to_filter_for: list = None,
    return_format="markdown-short",
) -> dict:
    """
    Build and execute a SPARQL query on Wikidata that retrieves adjacent properties, property labels,
    values, and value labels in the specified direction for a given entity.

    Args:
        entity: The entity (e.g., 'wd:Q187805') whose adjacent relations and entities we want to fetch.
        direction: The direction of the relation to consider. Can only be 'outgoing' or 'incoming'.
        properties_to_filter_for: Optional list of properties to filter for. If provided, only
                                 entities connected through these properties will be returned.

    Returns:
        A JSON object containing the query results.
    """

    if direction == "outgoing":
        # entity is subject
        triple_pattern = f"{entity} ?property ?value ."
    elif direction == "incoming":
        # entity is object
        triple_pattern = f"?value ?property {entity} ."
    else:
        raise ValueError("direction must be 'outgoing' or 'incoming'.")

    # Create VALUES clause if properties_to_filter_for is provided
    values_clause = ""
    if properties_to_filter_for and len(properties_to_filter_for) > 0:
        property_uris = []
        for prop in properties_to_filter_for:
            # Properties come in format like 'wdt:P31' or 'P31'
            prop = prop.strip()
            if prop.startswith("<") and prop.endswith(">"):
                # Already a full URI
                prop_uri = prop
            elif prop.startswith("wdt:"):
                # Format like 'wdt:P31' - convert to full URI
                prop_id = prop[4:]  # Remove 'wdt:' prefix
                prop_uri = f"<http://www.wikidata.org/prop/direct/{prop_id}>"
            elif prop.startswith("P") and prop[1:].isdigit():
                # Just the property ID like 'P31'
                prop_uri = f"<http://www.wikidata.org/prop/direct/{prop}>"
            else:
                # Fallback - assume it's a property ID
                prop_uri = f"<http://www.wikidata.org/prop/direct/{prop}>"
            property_uris.append(prop_uri)

        values_clause = f"VALUES ?property {{ {' '.join(property_uris)} }}"

    # We'll retrieve data in pages to avoid timeouts
    page_size = 10000  # Adjust as needed
    max_results = 50000
    total_results = 0
    offset = 0
    all_bindings = []  # we accumulate rows from each "page"

    while True:
        query = f"""
                PREFIX wd: <http://www.wikidata.org/entity/>
                PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

                SELECT DISTINCT ?property ?propertyLabel ?value ?valueLabel
                WHERE {{
                {values_clause}
                {triple_pattern}
                # FILTER(STRSTARTS(STR(?property), "http://www.wikidata.org/prop/direct/P"))
                OPTIONAL {{
                    ?property rdfs:label ?propLabel .
                    FILTER (lang(?propLabel) = "en")
                }}
                BIND(COALESCE(?propLabel, "") AS ?propertyLabel)

                OPTIONAL {{
                    ?value rdfs:label ?valLabel .
                    FILTER (lang(?valLabel) = "en")
                }}
                BIND(COALESCE(?valLabel, "") AS ?valueLabel)
                }}
                LIMIT {page_size}
                OFFSET {offset} 
                """

        page_results = execute_sparql_query(query, sparql_endpoint)

        if not page_results:
            break

        page_bindings = filter_non_english_rows(page_results)
        page_bindings = keep_only_direct_prop_rows_qald(page_bindings)
        page_bindings = fill_empty_property_labels(page_bindings, sparql_endpoint)
        page_bindings = shorten_property_and_value_uris_qald(page_bindings)

        all_bindings.extend(page_bindings)
        total_results += len(page_results)

        if len(page_results) < page_size or total_results >= max_results:
            break

        offset += page_size

    all_bindings = {
        "head": {"vars": ["property", "propertyLabel", "value", "valueLabel"]},
        "results": {"bindings": flatten_sparql_bindings(all_bindings)},
    }

    filtered_results = remove_duplicates_from_sparql_result(all_bindings)

    # Check if we have too many results and properties_to_filter_for was not provided
    if (
        len(filtered_results["results"]["bindings"]) > 50
        and not properties_to_filter_for
        and return_format == "markdown-short"
    ):
        # Extract unique properties and their labels
        unique_properties = {}
        for triple in filtered_results["results"]["bindings"]:
            prop = triple.get("property", "")
            prop_label = triple.get("propertyLabel", "")
            if prop and prop not in unique_properties:
                unique_properties[prop] = prop_label

        # print(f"unique_properties: {unique_properties}")
        unique_properties_table = convert_properties_to_table(unique_properties)
        # print(f"unique_properties_table:\n\n{unique_properties_table}")

        return unique_properties_table

    top_k = 4000
    if len(filtered_results["results"]["bindings"]) >= top_k:
        # Slice to first top_k results to avoid overload
        filtered_results["results"]["bindings"] = filtered_results["results"][
            "bindings"
        ][:top_k]

    if "markdown" in return_format:
        return convert_tool_result_to_table(filtered_results)
    elif "json" in return_format:
        return str(filtered_results)
        # return json.dumps(filtered_results, indent=0)

    raise ValueError("return_format must be 'markdown-short' or 'markdown' or 'json'.")

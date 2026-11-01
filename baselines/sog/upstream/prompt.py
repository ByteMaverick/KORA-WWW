TOOLS_WIKIDATA = [
    {
        "type": "function",
        "function": {
            "name": "get_adjacent_relations_and_entities",
            "description": (
                "Build and execute a SPARQL query on Wikidata that retrieves adjacent properties, property labels,"
                "values, and value labels in the specified direction for a given entity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "The entity (e.g., 'wd:Q187805') whose adjacent relations and entities we want to fetch.",
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["incoming", "outgoing"],
                        "description": "Direction of relationship to consider",
                    },
                    "properties_to_filter_for": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of specific properties to filter by (e.g., ['people.person.place_of_birth', 'people.person.nationality'])."
                    }
                },
                "required": ["question", "entity", "direction"],
                "additionalProperties": False,
            },
    
        },
    },
]


TOOLS_FREEBASE = [
    {
        "type": "function",
        "function": {
            "name": "get_adjacent_relations_and_entities",
            "description": (
                "Build and execute a SPARQL query on Freebase that retrieves adjacent properties, property labels,"
                "values, and value labels in the specified direction for a given entity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "The entity (e.g., 'm.04yd0fh') whose adjacent relations and entities we want to fetch."
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["incoming", "outgoing"],
                        "description": "Direction of relationship to consider"
                    },
                    "properties_to_filter_for": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of specific properties to filter by (e.g., ['people.person.place_of_birth', 'people.person.nationality'])."
                    }
                },
                "required": ["question", "entity", "direction"],
                "additionalProperties": False
            },
    
        }
    },
]


cwq_examples = [
"""
--
Question: What structure build on June 5, 2007 is Charlotte, North Carolina known for? {'Charlotte': 'm.0fsb8'}
'm.0fsb8' ('Charlotte') is the topic entity of the question. Look for outgoing edges from 'm.0fsb8' and look for relations and entities related to structures built in Charlotte. Any linked object entities are potential candidates.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.0fsb8", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
common.topic.topical_webpage|Topical webpage|http://www.charmeck.org/|
travel.travel_destination.tourist_attractions|Tourist attractions|m.09k6h_2|Bechtler Museum of Modern Art
travel.travel_destination.tourist_attractions|Tourist attractions|m.02vnhrq|Billy Graham Library
travel.travel_destination.tourist_attractions|Tourist attractions|m.05g_v0l|Bojangles' Coliseum
travel.travel_destination.tourist_attractions|Tourist attractions|m.0cq5c0|Carolinas Aviation Museum
We see the property travel.travel_destination.tourist_attractions (Tourist attractions), which points to m.09k6h_2 (Bechtler Museum of Modern Art), m.02vnhrq (Billy Graham Library), and m.05g_v0l (Bojangles' Coliseum). These are all tourist attractions in Charlotte, North Carolina.
Get outgoing relations and entities from each candidate entity to find information on the date that it was built:
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.09k6h_2", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
type.object.type|Type|common.topic|Topic
type.object.type|Type|architecture.building|Building
type.object.type|Type|architecture.structure|Structure
There is no property that indicates the build date of m.09k6h_2 (Bechtler Museum of Modern Art).
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.02vnhrq", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
common.topic.notable_types|Notable types|m.01y2hbz|Museum
architecture.structure.opened|Opened|2007-06-05-08:00|
type.object.type|Type|base.type_ontology.abstract|Abstract
We see that there is the property architecture.structure.opened (Opened), which points to the date 2007-06-05-08:00. This indicates an opening date of 2007-06-05 (June 5, 2007), which matches our target date.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.05g_v0l", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
architecture.structure.opened|Opened|1955-08:00|
common.topic.social_media_presence|Social media presence|http://www.facebook.com/pages/Bojangles-Coliseum/196122978761|
common.topic.social_media_presence|Social media presence|https://twitter.com/BojanglesCol|
We see that there is the property architecture.structure.opened (Opened), which points to the date 1955-08:00. This indicates an opening date of 1955 at 8am, which does not match our target date of June 5, 2007.
Final answer: Charlotte, North Carolina is known for the structure {Billy Graham Library} that is built on June 5, 2007.
""",
"""
--
Question: What did the actor from "Dallas Reunion: The Return to Southfork" die of? {'Dallas Reunion: The Return to Southfork': 'm.02z2yth'}
"m.02z2yth" ('Dallas Reunion: The Return to Southfork') is the topic entity of the question. Look for incoming edges from 'm.02z2yth' and look for relations and entities related to actors in Dallas Reunion: The Return to Southfork. Any linked object entities are potential candidates.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.02z2yth", "direction": "incoming"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
tv.tv_genre.programs|TV programs of this genre|m.03ch14|Television special
film.performance.film|Film|m.0j81my6|
tv.regular_tv_appearance.series|Series|m.04dcjbn|
We see that there is the property film.performance.film (Film), which links to the entity m.0j81my6.
Get incoming relations from m.0j81my6 (Film Peformance 1):
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.0j81my6", "direction": "incoming"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
type.type.instance|Instance|film.performance|Film performance
film.film.starring|Performances|m.02z2yth|Dallas Reunion: The Return to Southfork
film.actor.film|Film performances|m.01vyv9|Larry Hagman
We see that there is the property film.actor.film (Film Performances), which links to the entity m.01vyv9 (Harry Hagman). 
Get outgoing relations from m.01vyv9 (Harry Hagman) to find information on his cause of death:
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.01vyv9", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
people.deceased_person.cause_of_death|Cause of death|m.0369n2|Head and neck cancer
people.deceased_person.cause_of_death|Cause of death|m.04psf|Leukemia
people.deceased_person.cause_of_death|Cause of death|m.0qcr0|Cancer
We see that there is the property people.deceased_person.cause_of_death (Cause of death), which links to the entities m.0369n2 (Head and neck cancer), m.04psf (Leukemia), and m.0qcr0 (Cancer).
Final answer: The actor from "Dallas Reunion: The Return to Southfork" died of {Head and neck cancer}, {Leukemia}, and {Cancer}.
""",
"""
--
Question: What political party was home to the politician who was appointed by John Quincy Adams? {'John Quincy Adams': 'm.03_nq'}
"m.03_nq" ('John Quincy Adams') is the topic entity of the question. Look for incoming edges from 'm.03_nq' and look for relations and entities related to politicians appointed by John Quincy Adams. Any linked object entities are potential candidates.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.03_nq", "direction": "incoming"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
government.government_position_held.appointed_by|Appointed By (if Position is Appointed)|m.08v_t0x|
base.kwebbase.kwsentence.kwtopic|KWTopic|m.0c0t86y|Consequently, John Quincy was sent to Berlin as the Minister to Germany.
government.election_campaign.candidate|Candidate|m.04dnyqp|John Quincy Adams Presidential Campaign, 1824
We see that there is the property government.government_position_held.appointed_by (Appointed By (if Position is Appointed)), which links to the entity m.08v_t0x.
Get incoming relations from m.08v_t0x (Government Position 1):
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.08v_t0x", "direction": "incoming"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
government.politician.government_positions_held|Government Positions Held|m.0ct18|Henry Clay
government.government_office_or_title.office_holders|Office Holders|m.07y07|United States Secretary of State
government.political_appointer.appointees|Appointees|m.03_nq|John Quincy Adams
We see that there is the property government.politician.government_positions_held (Government Positions Held), which links to the entity m.0ct18 (Henry Clay).
Get outgoing relations from m.0ct18 (Henry Clay) to find information on the political party he was a member of:
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.0ct18", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
government.politician.party|Party|m.03lkh_7|
government.politician.party|Party|m.03lkh_2|
government.politician.party|Party|m.03lkhzy|
We see that there is the property government.politician.party (Party), which links to the entities m.03lkh_7, m.03lkh_2, and m.03lkhzy.
Get outgoing relations from candidate entities to get the names of each party:
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.03lkh_7", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
type.object.type|Type|government.political_party_tenure|Political Party Tenure
government.political_party_tenure.party|Party|m.07wf9|Democratic-Republican Party
government.political_party_tenure.politician|Politician|m.0ct18|Henry Clay
We see that there is the property government.political_party_tenure.party (Party), which links to the entity m.07wf9 (Democratic-Republican Party).
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.03lkh_2", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
type.object.type|Type|government.political_party_tenure|Political Party Tenure
government.political_party_tenure.party|Party|m.07wdw|Whig Party
government.political_party_tenure.politician|Politician|m.0ct18|Henry Clay
We see that there is the property government.political_party_tenure.party (Party), which links to the entity m.07wdw (Whig Party).
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.03lkhzy", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
type.object.type|Type|government.political_party_tenure|Political Party Tenure
government.political_party_tenure.party|Party|m.01spy8|National Republican Party
government.political_party_tenure.politician|Politician|m.0ct18|Henry Clay
We see that there is the property government.political_party_tenure.party (Party), which links to the entity m.01spy8 (National Republican Party).
Final answer: The political parties that were home to the politician who was appointed by John Quincy Adams are {Democratic-Republican Party}, {Whig Party}, and {National Republican Party}.
""",
"""
--
Question: Who wrote the music for The Golden Blaze and was also the voice of Simba? {'Voice': 'm.02nsjvf', 'The Golden Blaze': 'm.027tdtt', 'Simba': 'm.03wjyj'}
"m.03wjyj" ('Simba') is the topic entity of the question. Look for outgoing edges from 'm.03wjyj' and look for relations and entities related to films that portray Simba. Any linked object entities are potential candidates.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.03wjyj", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
film.film_character.portrayed_in_films|Portrayed in films|m.0nj0t6l|
film.film_character.portrayed_in_films|Portrayed in films|m.0gj592b|
film.film_character.portrayed_in_films|Portrayed in films|m.0gj592m|
film.film_character.portrayed_in_films|Portrayed in films|m.0y63lw5|
We see that there is the property film.film_character.portrayed_in_films (Portrayed in films) that links to the entities m.0nj0t6l, m.0gj592b, m.0gj592m, and m.0y63lw5.
Get outgoing relations from candidate entities to look for information about who is the actor (m.02nsjvf (Voice)) of Simba in those film performances and if that actor also wrote the music for m.027tdtt (The Golden Blaze):
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.0nj0t6l", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
film.performance.actor|Actor|m.01npcsf|Joseph Williams
film.performance.film|Film|m.099bhp|Mickey's Magical Christmas: Snowed in at the House of Mouse
film.performance.special_performance_type|Special Performance Type|m.02nsjvf|Voice
We see that there is the property film.performance.actor (Actor) that links to the entity m.01npcsf (Joseph Williams), and the property film.performance.special_performance_type (Special Performance Type) that links to the entity m.02nsjvf (Voice).
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.0gj592b", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
film.performance.actor|Actor|m.01vlj1g|Matthew Broderick
film.performance.film|Film|m.01sq_q|The Lion King II: Simba's Pride
film.performance.special_performance_type|Special Performance Type|m.02nsjvf|Voice
We see that there is the property film.performance.actor (Actor) that links to the entity m.01vlj1g (Matthew Broderick), and the property film.performance.special_performance_type (Special Performance Type) that links to the entity m.02nsjvf (Voice).
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.0gj592m", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
film.performance.actor|Actor|m.01vlj1g|Matthew Broderick
film.performance.film|Film|m.0224rs|The Lion King 1½
film.performance.special_performance_type|Special Performance Type|m.02nsjvf|Voice
We see that there is the property film.performance.actor (Actor) that links to the entity m.01vlj1g (Matthew Broderick), and the property film.performance.special_performance_type (Special Performance Type) that links to the entity m.02nsjvf (Voice).
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.0y63lw5", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
film.performance.actor|Actor|m.0nhtx9t|Evan Saucedo
film.performance.special_performance_type|Special Performance Type|m.02t8yb|Playback Singer
film.performance.special_performance_type|Special Performance Type|m.02yx9l4|Uncredited
We see that there is the property film.performance.actor (Actor) that links to the entity m.0nhtx9t (Evan Saucedo), and the property film.performance.special_performance_type (Special Performance Type) that links to the entities m.02t8yb (Playback Singer) and m.02yx9l4 (Uncredited).
Get outgoing relations from candidate entities to find information on music they have written
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.01npcsf", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
film.music_contributor.film|Film music credits|m.04j1brz|Written in Blood
film.music_contributor.film|Film music credits|m.026sj6d|Judgment Day
film.music_contributor.film|Film music credits|m.027tdtt|The Golden Blaze
We see that there is the property film.music_contributor.film (Film music credits) that links to the entities m.04j1brz (Written in Blood), m.026sj6d (Judgment Day), and m.027tdtt (The Golden Blaze).
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.01vlj1g", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
theater.theater_actor.theater_roles|Theater Roles|m.05v6j8j|
theater.theater_actor.theater_roles|Theater Roles|m.05v6j4h|
type.object.type|Type|film.person_or_entity_appearing_in_film|Person or entity appearing in film
Final answer: The person who wrote the music for The Golden Blaze and was also the voice of Simba is {Joseph Williams}.
""",
"""
--
Question: What college Brian Williams is a university with more than 10505 enrolled undergraduates? {'Brian Williams': 'm.030p0z', 'College/University': 'm.01y2hnl'}
"m.030p0z" ('Brian Williams') is the topic entity of the question. Look for outgoing edges from 'm.030p0z' and look for relations and entities related to colleges that m.030p0z (Brian Williams) attended. Any linked object entities are potential candidates.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.030p0z", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
people.person.education|Education|m.03p8npy|
people.person.education|Education|m.0fs7k58|
people.person.education|Education|m.02wnzrb|
We see that there is the property people.person.education (Education) that links to the entities m.03p8npy, m.0fs7k58, and m.02wnzrb.
Get outgoing relations from candidate entities and look for information regarding the name of each education institution
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.03p8npy", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
education.education.institution|Institution|m.01nnsv|George Washington University
education.education.student|Student|m.030p0z|Brian Williams
We see that there is the property education.education.institution (Institution) that links to the entity m.01nnsv (George Washington University).
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.0fs7k58", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
type.object.type|Type|education.education|Education
education.education.institution|Institution|m.05twn6|Brookdale Community College
We see that there is the property education.education.institution (Institution) that links to the entity m.05twn6 (Brookdale Community College).
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.02wnzrb", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
type.object.type|Type|education.education|Education
education.education.institution|Institution|m.02vnp2|The Catholic University of America
We see that there is the property education.education.institution (Institution) that links to the entity m.02vnp2 (The Catholic University of America).
Get outgoing relations from candidate entities and look for information regarding whether or not that instution is indeed a university (m.01y2hnl), and its number of enrolled undergraduates:
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.01nnsv", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
common.topic.notable_types|Notable types|m.01y2hnl|College/University
education.university.number_of_undergraduates|Number of undergraduates|m.0zrjd_w|
education.educational_institution.students_graduates|Students/Graduates|m.0svys_d|
We see that there is the property common.topic.notable_types (Notable types) that links to the entity m.01y2hnl (College/University), and the property education.university.number_of_undergraduates (Number of undergraduates) that links to the entity m.0zrjd_w. So, we look at outgoing relations from m.0zrjd_w (Number of undergraduates) to find the value of ?num.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.0zrjd_w", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
measurement_unit.dated_integer.number|number|10464|
We see that there is the property measurement_unit.dated_integer.number (number) that links to the entity 10464.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.05twn6", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
common.topic.notable_types|Notable types|m.01y2hnl|College/University
education.university.number_of_undergraduates|Number of undergraduates|m.0h77vcn|
We see that there is the property common.topic.notable_types (Notable types) that links to the entity m.01y2hnl (College/University), and the property education.university.number_of_undergraduates (Number of undergraduates) that links to the entity m.0h77vcn. So, we look at outgoing relations from m.0h77vcn (Number of undergraduates) to find the value of ?num.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.0h77vcn", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
measurement_unit.dated_integer.number|number|15639|
We see that there is the property measurement_unit.dated_integer.number (number) that links to the entity 15639.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.02vnp2", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
common.topic.notable_types|Notable types|m.01y2hnl|College/University
education.university.number_of_undergraduates|Number of undergraduates|m.0k7fxj0|
We see that there is the property common.topic.notable_types (Notable types) that links to the entity m.01y2hnl (College/University), and the property education.university.number_of_undergraduates (Number of undergraduates) that links to the entity m.0k7fxj0. So, we look at outgoing relations from m.0k7fxj0 (Number of undergraduates) to find the value of ?num.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.0k7fxj0", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
measurement_unit.dated_integer.number|number|3573|
We see that there is the property measurement_unit.dated_integer.number (number) that links to the entity 3573.
Final answer: A college that Brian Williams attended and is a university with more than 10505 enrolled undergraduates is {Brookdale Community College}.
"""
]


grailqa_examples = [
"""
--
Question: what is the language regulator of basque? {'basque': 'm.017k6'}
'm.017k6' ('basque') is the topic entity of the question. Look for incoming edges from 'm.017k6' and look for relations and entities related to language regulators of Basque. Any linked object entities are potential candidates.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.017k6", "direction": "incoming"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
base.rosetta.rosetta_document.refers_to|Refers To|m.05tr3c6|Basque Numbers
language.language_regulator.language|Language|m.057xsn|Euskaltzaindia
type.type.instance|Instance|language.languoid|
We see the property language.language_regulator.language (Language), which points to m.057xsn (Euskaltzaindia). This may be the language regulator of Basque. Let's double check by calling the tool to look at its outgoing edges.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.057xsn", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
type.object.type|Type|common.topic|Topic
type.object.type|Type|base.type_ontology.agent|Agent
type.object.type|Type|language.language_regulator|Language Regulator
We see that there is the property type.object.type (Type), which points to language.language_regulator (Language Regulator). This confirms that m.057xsn (Euskaltzaindia) is indeed a language regulator.
Final answer: The language regulator of Basque is {Euskaltzaindia}.
""",
"""
--
Question: the albums confessions tour is part of what parent genre of a musical genre? {'confessions tour': 'm.03fz6lg'}
'm.03fz6lg' ('confessions tour') is the topic entity of the question. Look for incoming edges from 'm.03fz6lg' and look for relations and entities related to its musical genres. Any linked object entities are potential candidates.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.03fz6lg", "direction": "incoming"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
music.release.album|Release of|m.0f2dhj2|The Confessions Tour
music.genre.albums|Albums|m.06by7|Rock music
music.genre.albums|Albums|m.064t9|Pop music
We see that there is the property music.genre.albums (Albums) that links to the entities m.06by7 (Rock music) and m.064t9 (Pop music). These are musical genres of The Confessions Tour. Let's look at the outgoing edges of these musical genres for relations and entities related to their parent genre.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.06by7", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
music.genre.parent_genre|Parent genres|m.02w4v|Folk music
broadcast.radio_format.stations|Stations|m.0h19gj|WRKT
music.genre.subgenre|Subgenres|m.0137lv|Pub rock
We see that there is the property music.genre.parent_genre (Parent genres), which points to m.02w4v (Folk music). This means that Folk music is the parent genre of Rock music.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.064t9", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
music.genre.subgenre|Subgenres|m.0w1cxlb|B-POP
music.genre.parent_genre|Parent genres|m.0ggx5q|Dance music
music.genre.parent_genre|Parent genres|m.0gpjmr|Russian pop
We see that there is the property music.genre.parent_genre (Parent genres) that links to the entities m.0ggx5q (Dance music) and m.0gpjmr (Russian pop). These are parent genres of Pop music.
Final answer: The Confessions Tour is part of the parent genres {Folk music}, {Dance music}, and {Russian pop} of musical genres.
""",
"""
--
Question: what is the most recent musical release that credits engineers including lee whitsel? {'Lee Whitsel': 'm.04t59_w'}
'm.04t59_w' ('Lee Whitsel') is the topic entity of the question. Look for incoming edges from 'm.04t59_w' and look for relations and entities related to musical releases that credit Lee Whitsel as an engineer. Any linked object entities are potential candidates.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.04t59_w", "direction": "incoming"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
music.release.engineers|Engineers|m.0dl5wzp|Witnesses
music.release.engineers|Engineers|m.0dl5xf1|Continuity
music.release.engineers|Engineers|m.0bflrzl|Still Unsatisfied
type.type.instance|Instance|music.writer|
We see that there is the property music.release.engineers (Engineers) that links to the entities m.0dl5wzp (Witnesses), m.0dl5xf1 (Continuity), and m.0bflrzl (Still Unsatisfied). These are musical releases that credit Lee Whitsel as an engineer. Let's look at the outgoing edges of these musical releases for relations and entities related to their release dates.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.0dl5wzp", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
type.object.type|Type|common.topic|Topic
common.topic.notable_types|Notable types|m.01xlj26|Musical Release
music.release.album|Release of|m.0cr6t06|Witnesses
We do not see any property related to the release date of Witnesses. Let's check the next musical release.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.0dl5xf1", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
media_common.creative_work.credit|Credited As|Wedlock|
music.release.catalog_number|Catalog Number|KOU08|
music.release.release_date|Release Date|2009-10-27-08:00|
We see that there is the property music.release.release_date (Release Date), which points to 2009-10-27-08:00. This means that Continuity was released on October 27, 2009.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.0bflrzl", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
type.object.type|Type|base.type_ontology.non_agent|Non-Agent
common.topic.notable_types|Notable types|m.01xlj26|Musical Release
music.release.engineers|Engineers|m.04t59_w|Lee Whitsel
We do not see any property related to the release date of Still Unsatisfied. Therefore, the most recent musical release that credits engineers including Lee Whitsel is Continuity, which was released on October 27, 2009.
Final answer: The most recent musical release that credits engineers including Lee Whitsel is {Continuity}.
""",
"""
--
Question: what was the subject of the tv show with the most number of episodes and featured on killer joke? {'killer joke': 'm.04vd3'}
'm.04vd3' ('killer joke') is the topic entity of the question. Look for outgoing edges from 'm.04vd3' and look for relations and entities related to the subject of the TV shows featured on Killer Joke. Any linked object entities are potential candidates.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.04vd3", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
common.topic.notable_types|Notable types|m.0606t7p|TV episode segment
tv.tv_episode_segment.episodes|Episodes|m.03bxq_s|Whither Canada
tv.tv_episode_segment.subjects|Subjects|m.03nn8ps|Humor theory
tv.tv_episode_segment.subjects|Subjects|m.081pw|World War II
We see that there is the property tv.tv_episode_segment.subjects (Subjects) that links to the entities m.03nn8ps (Humor theory) and m.081pw (World War II). These are subjects of TV shows featured on Killer Joke. Let's look at the outgoing edges of these subjects for relations and entities related to the number of episodes.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.03nn8ps", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
type.object.type|Type|tv.tv_subject|TV subject
type.object.type|Type|common.topic|Topic
tv.tv_subject.tv_episode_segments|TV episode segments|m.04vd3|The Funniest Joke in the World
We do not see any property related to the number of episodes for Humor theory. Let's check the next subject.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.081pw", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
military.military_conflict.military_personnel_involved|Military personnel involved|m.02qwh98|Franz Machon
tv.tv_subject.tv_programs|TV programs|m.09_k1t|Anne Frank: The Whole Story
tv.tv_subject.tv_programs|TV programs|m.015g28|Band of Brothers
cvg.computer_game_subject.games|Games On This Subject|m.020594|Call of Duty
We see that there is the property tv.tv_subject.tv_programs (TV programs) that links to the entities m.09_k1t (Anne Frank: The Whole Story) and m.015g28 (Band of Brothers). These are TV programs about World War II. Let's look at the outgoing edges of these TV programs for relations and entities related to the number of episodes.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.09_k1t", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
tv.tv_program.episode_running_time|Original episode running time|189|
tv.tv_program.number_of_episodes|Number of episodes|2|
tv.tv_program.air_date_of_final_episode|Air date of final episode|2001-05-21-08:00|
We see that there is the property tv.tv_program.number_of_episodes (Number of episodes), which points to 2. This means that Anne Frank: The Whole Story has 2 episodes.
Therefore, the subject of the TV show with the most number of episodes and featured on Killer Joke is {World War II}.
Final answer: The subject of the TV show with the most number of episodes and featured on Killer Joke is {World War II}.
""",
"""
--
Question: how many ice hockey coaches have the boston junior bruins had? {'Boston Junior Bruins': 'm.0413knt'}
'm.0413knt' ('Boston Junior Bruins') is the topic entity of the question. Look for incoming edges from 'm.0413knt' and look for relations and entities related to ice hockey coaches of the Boston Junior Bruins. Any linked object entities are potential candidates.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.0413knt", "direction": "incoming"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
ice_hockey.hockey_coach.current_team|Current Team Coached|m.0j0xwb1|Chris Masters
ice_hockey.hockey_coach.current_team|Current Team Coached|m.0j0xwb7|Peter Masters
sports.sports_facility.teams|Teams|m.0bbzkz9|New England Sports Center
We see that there is the property ice_hockey.hockey_coach.current_team (Current Team Coached) that links to the entities m.0j0xwb1 (Chris Masters) and m.0j0xwb7 (Peter Masters). They may be ice hockey coaches of the Boston Junior Bruins. Let's double check by calling the tool to look at their outgoing edges.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.0j0xwb1", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
type.object.type|Type|ice_hockey.hockey_coach|Ice hockey coach
type.object.type|Type|common.topic|Topic
common.topic.notable_types|Notable types|m.029fbpr|Ice hockey coach
We see that there is the property type.object.type (Type), which points to ice_hockey.hockey_coach (Ice hockey coach). This confirms that m.0j0xwb1 (Chris Masters) is indeed an ice hockey coach.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.0j0xwb7", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
type.object.type|Type|ice_hockey.hockey_coach|Ice hockey coach
type.object.type|Type|common.topic|Topic
common.topic.notable_types|Notable types|m.029fbpr|Ice hockey coach
We see that there is the property type.object.type (Type), which points to ice_hockey.hockey_coach (Ice hockey coach). This confirms that m.0j0xwb7 (Peter Masters) is indeed an ice hockey coach.
Therefore, the Boston Junior Bruins have had 2 ice hockey coaches: Chris Masters and Peter Masters.
Final answer: The Boston Junior Bruins have had {2} ice hockey coaches.
"""
]
webqsp_examples = [
"""
--
Question: what is cher 's son 's name {'Cher': 'm.01vtj38', 'Male': 'm.05zppz'}
'm.01vtj38' ('Cher') is the topic entity of the question. Look for outgoing edges from 'm.01vtj38' and look for relations and entities related to Cher's son. Any linked object entities are potential candidates.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.01vtj38", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
base.schemastaging.context_name.official_name|Official name|Cherilyn Sarkisian|
people.person.children|Children|m.01w4bt1|Elijah Blue Allman
people.person.children|Children|m.0br66|Chaz Bono
people.person.parents|Parents|m.0kmhsk2|Gilbert Hartmann LaPiere
We see there is the people.person.children (Children) property that points to entities m.01w4bt1 (Elijah Blue Allman) and m.0br66 (Chaz Bono).
To specifically look for son(s) of Cher, we check the gender of each of those entities
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.01w4bt1", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
people.person.parents|Parents|m.01vtj38|Cher
people.person.gender|Gender|m.05zppz|Male
people.person.sibling_s|Siblings|m.0w4gdrb|
people.person.sibling_s|Siblings|m.0vvfthw|
We see that Elijah Blue Allman's gender is Male.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.0br66", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
people.person.parents|Parents|m.01vtj38|Cher
people.person.sibling_s|Siblings|m.0w4gdrb|
people.person.gender|Gender|m.05zppz|Male
We see that Chaz Bono's gender is Male.
Final answer: Cher's sons' names are {Elijah Blue Allman} and {Chaz Bono}.
""",
"""
--
Question: what country is russia close to {'Russia': 'm.06bnz', 'Country': 'm.01mp'}
'm.06bnz' ('Russia') is the topic entity of the question. Look for outgoing edges from 'm.06bnz' and look for relations and entities related to countries close to Russia. Any linked entities are potential candidates.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.06bnz", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
location.location.contains|Contains|g.122zkbns|
kg.object_profile.prominent_type||location.country|Country
location.location.adjoin_s|Adjoins|m.02xw90z|
location.location.adjoin_s|Adjoins|m.05gq88m|
location.location.adjoin_s|Adjoins|m.05g_659|
type.object.type|Type|location.country|Country
base.biblioness.bibs_location.loc_type|Type|Country|
We see that there is the property location.location.adjoin_s (Adjoins) that points to the entities m.02xw90z, m.05gq4vn, m.05g_659.
Try to find the name of each of those entities by exploring their outgoing relations and entities.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.02xw90z", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
type.object.type|Type|location.adjoining_relationship|Adjoining Relationship
location.adjoining_relationship.adjoins|Adjoins|m.05b4w|Norway
We see that there is the property location.adjoining_relationship.adjoins (Adjoins) that points to m.05b4w (Norway).
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.05gq4vn", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
type.object.type|Type|location.adjoining_relationship|Adjoining Relationship
location.adjoining_relationship.adjoins|Adjoins|m.0jhd|Azerbaijan
location.adjoining_relationship.adjoins|Adjoins|m.06bnz|Russia
We see that there is the property location.adjoining_relationship.adjoins (Adjoins) that points to m.0jhd (Azerbaijan).
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.05g_659", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
type.object.type|Type|location.adjoining_relationship|Adjoining Relationship
location.adjoining_relationship.adjoins|Adjoins|m.02kmm|Estonia
location.adjoining_relationship.adjoins|Adjoins|m.06bnz|Russia
We see that there is the property location.adjoining_relationship.adjoins (Adjoins) that points to m.02kmm (Estonia).
Look one hop further from each of those entities to verify that they are a country
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.05b4w", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
common.topic.notable_types|Notable types|m.01mp|Country
common.topic.notable_for|Notable for|g.1256fvfg4|
biology.breed_origin.breeds_originating_here|Breeds orginating here|m.027dd3t|Black Norwegian Elkhound
We see that there is the property common.topic.notable_types (Notable types) that points to m.01mp (Country).
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.0jhd", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
location.location.partiallycontains|Partially contains|m.0wg91m4|
location.location.partiallycontains|Partially contains|m.0wg91k6|
common.topic.notable_types|Notable types|m.01mp|Country
We see that there is the property common.topic.notable_types (Notable types) that points to m.01mp (Country).
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.02kmm", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
location.location.people_born_here|People born here|g.122c64rf|Mikhail Volkonsky
olympics.olympic_participating_country.athletes|Athletes|m.0khq2wv|
common.topic.notable_types|Notable types|m.01mp|Country
We see that there is the property common.topic.notable_types (Notable types) that points to m.01mp (Country).
Final answer: Russia is close to {Norway}, {Azerbaijan}, {Estonia}.
""",
"""
--
Question: what is caesar augustus known for {'Augustus': 'm.0ngg'}
'm.0ngg' ('Augustus') is the topic entity of the question. Look for outgoing edges from 'm.0ngg' and look for relations and entities related to what Caesar Augustus is known for. Any linked entities are potential candidates.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.0ngg", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
type.object.type|Type|base.adoption.adopted_person|Adopted person
people.person.profession|Profession|m.0fj9f|Politician
people.person.profession|Profession|m.01g14w|Roman emperor
We see that there is the property people.person.profession (Profession) that points to m.0fj9f (Politician) and m.01g14w (Roman emperor).
Final answer: Caesar Augustus is known for being a {Politician} and {Roman emperor}.
""",
"""
--
Question: who voices Darth Vader in the clone wars {'Darth Vader': 'm.0f2y0', 'Star Wars: The Clone Wars': 'm.0cpllql'}
'm.0cpllql' ('Star Wars: The Clone Wars') is the topic entity of the question. Look for outgoing edges from 'm.0cpllql' and look for relations and entities related to the person who voices Darth Vader. Any linked object entities are potential candidates.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.0cpllql", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
film.film.other_crew|Other crew|m.0bvhvs2|
film.film.starring|Performances|m.0bvhs3r|
film.film.starring|Performances|m.0bvhs6d|
film.film.starring|Performances|m.04m6bsf|
We see the property film.film.starring (Performances) that links to entities m.0bvhs3r, m.0bvhs6d, and m.04m6bsf.
Explore one hop further from each of those entities to look for information related to Darth Vader
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.0bvhs3r", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
type.object.type|Type|film.performance|Film performance
film.performance.actor|Actor|m.0582cf|Dee Bradley Baker
film.performance.film|Film|m.0cpllql|Star Wars: The Clone Wars
film.performance.character|Character|g.120vyyqj|Cody
film.performance.special_performance_type|Special Performance Type|m.02nsjvf|Voice
We see that there is a film.performance.character (Character) property, which links to g.120vyyqj (Cody).
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.0bvhs6d", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
type.object.type|Type|film.performance|Film performance
film.performance.actor|Actor|m.01yh3y|Corey Burton
film.performance.film|Film|m.0cpllql|Star Wars: The Clone Wars
film.performance.character|Character|m.0bvhs6h|Ziro the Hutt
film.performance.special_performance_type|Special Performance Type|m.02nsjvf|Voice
We see that there is a film.performance.character (Character) property, which links to m.0bvhs6h (Ziro the Hutt).
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.04m6bsf", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
type.object.type|Type|film.performance|Film performance
film.performance.actor|Actor|m.0b2l8f|Matt Lanter
film.performance.film|Film|m.0cpllql|Star Wars: The Clone Wars
film.performance.character|Character|m.0f2y0|Darth Vader
film.performance.special_performance_type|Special Performance Type|m.02nsjvf|Voice
We see that there is a film.performance.character (Character) property, which links to m.0f2y0 (Darth Vader). Since Darth Vader is exactly what is asked for in the question, we can get information regarding the voice actor of Darth Vader from the film.performance.actor property, which links to m.0b2l8f (Matt Lanter).
Final answer: The person who voices Darth Vader in "Star Wars: The Clone Wars" is {Matt Lanter}.
""",
"""
--
Question: what years have the dallas cowboys won the superbowl {'Dallas Cowboys': 'm.02896'}
'm.02896' ('Dallas Cowboys') is the topic entity of the question. Look for outgoing edges from 'm.02896' and look for relations and entities related to Super Bowl championships they won. Any linked object entities are potential candidates.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.02896", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
sports.sports_team.sport|Sport|m.0jm_|American football
sports.professional_sports_team.owner_s|Owner(s)|m.016b9c|Jerry Jones
sports.sports_team.championships|Championships|m.076vy|Super Bowl XXVIIsports.sports_team.championships|Championships|m.076q3|Super Bowl XII
sports.sports_team.championships|Championships|m.076w8|Super Bowl XXVIII
sports.sports_team.championships|Championships|m.076n1|Super Bowl VI
sports.sports_team.championships|Championships|m.076wz|Super Bowl XXX
We see that there is a sports.sports_team.championships (Championships) property, which links to m.076vy (Super Bowl XXVII), m.076q3 (Super Bowl XII), m.076w8 (Super Bowl XXVIII), m.076n1 (Super Bowl VI), and m.076wz (Super Bowl XXX).
Final answer: The Dallas Cowboys won the Super Bowl in the years {Super Bowl XXVII}, {Super Bowl XII}, {Super Bowl XXVIII}, {Super Bowl VI}, and {Super Bowl XXX}.
"""
]


simplequestions_examples = [
"""
--
Question: where did the continental celtic languages originate? {'Continental Celtic languages': 'm.06v3q8'}
Outgoing relations from m.06v3q8 (Continental Celtic languages)
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.06v3q8", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
language.language_family.member_of_language_families|member of language families|m.01sd8|Celtic languages
language.language_family.geographic_distribution|geographic distribution|m.02j9z|Europe
kg.object_profile.prominent_type||language.language_family|Language Family
We see there is a language.language_family.geographic_distribution (geographic distribution) property that points to the object m.02j9z (Europe).
Final answer: The Continental Celtic languages originated in {Europe}.
""",
"""
--
Question: what is the topic of the book teen idol? {'Teen Idol': 'm.0gv58g'}
Outgoing relations from m.0gv58g (Teen Idol)
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.0gv58g", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
book.book.genre|Genre|m.03mfnf|Young-adult fiction
book.written_work.subjects|Subjects|m.02jfhn_|Schools
book.written_work.subjects|Subjects|m.03v1s|Indiana
We see that there is a book.written_work.subjects (Subjects) property that points to the objects m.02jfhn_ (Schools) and m.03v1s (Indiana).
Final answer: The topic of the book Teen Idol is {Schools} and {Indiana}.
""",
"""
--
Question: What kind of materials were taken from the san josé mine {'San José Mine': 'm.0dgqd_c'}
Outgoing relations from m.0dgqd_c (San José Mine)
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.0dgqd_c", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
base.engineering.mine.materials_extracted|Materials Extracted|m.025sf8x|Silver
base.engineering.mine.materials_extracted|Materials Extracted|m.025rs2z|Gold
kg.object_profile.prominent_type||location.location|Location
We see that there is a base.engineering.mine.materials_extracted (Materials Extracted) property that points to the objects m.025sf8x (Silver) and m.025rs2z (Gold).
Final answer: The materials taken from the San José Mine are {Silver} and {Gold}.
""",
"""
--
Question: what hockey player is a centerman {'Centerman': 'm.02qvdc'}
Outgoing relations from m.02qvdc (Centerman)
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.02qvdc", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
ice_hockey.hockey_position.players|Players|m.019f2|Bernie Federko
ice_hockey.hockey_position.players|Players|m.0fnxsj|Darryl Laplante
ice_hockey.hockey_position.players|Players|m.04l04h|Tuomo Ruutu
ice_hockey.hockey_position.players|Players|m.0j8x64w|Louis-Marc Aubry
ice_hockey.hockey_position.players|Players|m.02rdb30|Bob Fitchner
ice_hockey.hockey_position.players|Players|m.03d2t5r|Vladimír Sobotka
ice_hockey.hockey_position.players|Players|m.03m68p|Greg Johnson
ice_hockey.hockey_position.players|Players|m.03c48ln|Shayne Stevenson
ice_hockey.hockey_position.players|Players|m.02rbbf6|Fred Hergerts
ice_hockey.hockey_position.players|Players|m.07b_l_|Shawn Horcoff
ice_hockey.hockey_position.players|Players|m.0gz34j|Mike Stapleton
ice_hockey.hockey_position.players|Players|m.0409698|Robin Lindqvist
ice_hockey.hockey_position.players|Players|m.0gvq_zy|Ben Street
We see that there is an ice_hockey.hockey_position.players (Players) property that points to the objects m.019f2 (Bernie Federko), m.0fnxsj (Darryl Laplante), m.04l04h (Tuomo Ruutu), m.0j8x64w (Louis-Marc Aubry), m.02rdb30 (Bob Fitchner), m.03d2t5r (Vladimír Sobotka), m.03m68p (Greg Johnson), m.03c48ln (Shayne Stevenson), m.02rbbf6 (Fred Hergerts), m.07b_l_ (Shawn Horcoff), m.0gz34j (Mike Stapleton), m.0409698 (Robin Lindqvist), m.0gvq_zy (Ben Street).
Final answer: The hockey players who are centermen include {Bernie Federko}, {Darryl Laplante}, {Tuomo Ruutu}, {Louis-Marc Aubry}, {Bob Fitchner}, {Vladimír Sobotka}, {Greg Johnson}, {Shayne Stevenson}, {Fred Hergerts}, {Shawn Horcoff}, {Mike Stapleton}, {Robin Lindqvist}, {Ben Street}.
""",
"""
--
Question: which asian language is spoken in tora tora tora gold vol.88 {'Tora Tora Tora Gold vol.88': 'm.0dcd0lm'}
Outgoing relations from m.0dcd0lm (Tora Tora Tora Gold vol.88)
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "m.0dcd0lm", "direction": "outgoing"}}}]
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
common.topic.notable_types|Notable types|m.0jsg2m|Film
film.film.country|Country of origin|m.03_3d|Japan
film.film.language|Languages|m.03_9r|Japanese Language
We see that there is a film.film.language (Languages) property that points to the object m.03_9r (Japanese Language).
Final answer: The asian language spoken in Tora Tora Tora Gold vol.88 is {Japanese Language}.
"""
]


qald_examples = [
"""
--
Question: What was the last movie with Alec Guinness? {'Alec Guinness': 'wd:Q103894', 'film': 'wd:Q11424'}wd:Q103894' ('Alec Guinness') is the topic entity of the question. Look for incoming edges to 'wd:Q103894' and look for relations and entities related to Alec Guinness' last movie. Any linked subjects are potential candidates.
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "wd:Q103894", "direction": "incoming"}}}]        
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
wdt:P1346|winner|wd:Q2925701|British Academy Television Award for Best Actor
wdt:P161|cast member|wd:Q181795|Star Wars: Episode V – The Empire Strikes Back
wdt:P161|cast member|wd:Q181803|Star Wars: Episode VI – Return of the Jedi
wdt:P161|cast member|wd:Q6074|Star Wars: Episode VII – The Force Awakens
We see the property wdt:P161 (cast member) that links to multiple entities, each with its own ID (e.g. wd:Q181795 (Star Wars: Episode V – The Empire Strikes Back), wd:Q181803 (Star Wars: Episode VI – Return of the Jedi), wd:Q6074 (Star Wars: Episode VII – The Force Awakens)).
Check that each entity is indeed a film and look for its release date by looking at their outgoing relations and entities:
Tool Call:[{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "wd:Q181795", "direction": "outgoing"}}}]        
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
wdt:P31|instance of|wd:Q11424|film
wdt:P2631|TCM Movie Database film ID|74051|
wdt:P577|publication date|1980-05-17T00:00:00Z|
We see the property wdt:P31 (instance of) that links to the entity wd:Q11424 (film). We also see the property wdt:P577 (publication date) that links to a date entity (e.g. 1980-05-17T00:00:00Z).
Tool Call:[{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "wd:Q181803", "direction": "outgoing"}}}]        
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
wdt:P344|director of photography|wd:Q958911|Alec Mills
wdt:P31|instance of|wd:Q11424|film
wdt:P577|publication date|1983-12-09T00:00:00Z|
We see the property wdt:P31 (instance of) that links to the entity wd:Q11424 (film). We also see the property wdt:P577 (publication date) that links to a date entity (e.g. 1983-12-09T00:00:00Z).
Tool Call:[{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "wd:Q6074", "direction": "outgoing"}}}]        
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
wdt:P2163|FAST ID|1938147|
wdt:P577|publication date|2015-12-18T00:00:00Z|
wdt:P31|instance of|wd:Q11424|film
We see the property wdt:P31 (instance of) that links to the entity wd:Q11424 (film). We also see the property wdt:P577 (publication date) that links to a date entity (e.g. 2015-12-18T00:00:00Z).
Identify the movie that has the latest publication date
The publication dates are:
- Star Wars: Episode V – The Empire Strikes Back: 1980-05-17T00:00:00Z
- Star Wars: Episode VI – Return of the Jedi: 1983-12-09T00:00:00Z
- Star Wars: Episode VII – The Force Awakens: 2015-12-18T00:00:00Z
The latest publication date is 2015-12-18T00:00:00Z, which corresponds to the movie Star Wars: Episode VII – The Force Awakens.
Final Answer: The last movie with Alec Guinness is {Star Wars: Episode VII – The Force Awakens}.
""",
"""
--
Question: In which countries do people speak Japanese? {'country': 'wd:Q6256', 'Japanese': 'wd:Q5287'}wd:Q6256' ('country') is the topic entity of the question. Look for incoming edges to 'wd:Q6256' and look for relations and entities related to countries where people speak Japanese. Any linked subjects are potential candidates.
Tool Call:[{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "wd:Q6256", "direction": "incoming"}}}]        
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
wdt:P31|instance of|wd:Q17|Japan
wdt:P360|is a list of|wd:Q2571596|
wdt:P31|instance of|wd:Q884|South Korea
wdt:P31|instance of|wd:Q695|Palau
We see the property wdt:P31 (instance of) that links to multiple entities, each with its own ID (e.g. wd:Q17 (Japan), wd:Q884 (South Korea), wd:Q695 (Palau)).
Look at each candidate entity's outgoing relations for information regarding its language spoken:
Tool Call:[{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "wd:Q17", "direction": "outgoing"}}}]        
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
wdt:P172|ethnic group|wd:Q101828|Ainu people
wdt:P37|official language|wd:Q5287|Japanese
wdt:P150|contains the administrative territorial entity|wd:Q81863|Akita Prefecture
We see the property wdt:P37 (official language) that links to the entity wd:Q5287 (Japanese).
Tool Call:[{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "wd:Q884", "direction": "outgoing"}}}]        
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
wdt:P37|official language|wd:Q3073428|Korean Sign Language
wdt:P12749|SNARC ID|wd:Q1334|
wdt:P31|instance of|wd:Q1520223|constitutional republic
We see the property wdt:P37 (official language) that links to the entity wd:Q3073428 (Korean Sign Language). Therefore, wdt:P37 (official language) does not link to entity wd:Q5287 (Japanese).
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "wd:Q695", "direction": "outgoing"}}}]        
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
wdt:P37|official language|wd:Q1860|English
wdt:P463|member of|wd:Q186444|Treaty on the Non-Proliferation of Nuclear Weapons
wdt:P37|official language|wd:Q5287|Japanese
We see the property wdt:P37 (official language) that links to the entities wd:Q5287 (Japanese) and wd:Q1860 (English).
Final Answer: The countries where people speak Japanese are {Japan} and {Palau}.
""",
"""
--
Question: Give me the currency of China. {"People's Republic of China": 'wd:Q148'}wd:Q148' ('People's Republic of China') is the topic entity of the question. Look for outgoing edges from 'wd:Q148' and look for relations and entities related to China's currency. Any linked subjects are potential candidates.
Tool Call:[{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "wd:Q148", "direction": "outgoing"}}}]        
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
wdt:P150|contains the administrative territorial entity|wd:Q11725|Chongqing
wdt:P150|contains the administrative territorial entity|wd:Q11736|Tianjin
wdt:P38|currency|wd:Q39099|renminbi
We see the property wdt:P38 (currency) that links to the entity wd:Q39099 (renminbi).
Final Answer: The currency of China is {renminbi}.
""",
"""
--
Question: In which country does the Ganges start? {'Ganges': 'wd:Q5089'}wd:Q5089' ('Ganges') is the topic entity of the question. Look for outgoing edges from 'wd:Q5089' and look for relations and entities related to which country the Ganges starts in. Any linked subjects are potential candidates.
Tool Call:[{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "wd:Q5089", "direction": "outgoing"}}}]        
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
wdt:P885|origin of the watercourse|wd:Q691557|Gangotri Glacier
wdt:P974|tributary|wd:Q3635865|Punpun River
wdt:P30|continent|wd:Q48|Asia
We see the property wdt:P885 (origin of the watercourse) that links to the entity wd:Q691557 (Gangotri Glacier).
Look at each candidate entity's outgoing relations for information regarding its country
Tool Call: [{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "wd:Q691557", "direction": "outgoing"}}}]        
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
wdt:P4552|mountain range|wd:Q3777888|Gangotri Group
wdt:P31|instance of|wd:Q35666|glacier
wdt:P17|country|wd:Q668|India
We see the property wdt:P17 (country) that links to the entity wd:Q668 (India).
Final Answer: The Ganges starts in {India}.
""","""--
Question: How many films did Hal Roach produce? {'Hal Roach': 'wd:Q72792'}wd:Q72792' ('Hal Roach') is the topic entity of the question. Look for incoming edges to 'wd:Q72792' and look for relations and entities related to films produced by Hal Roach. Any linked subjects are potential candidates.
Tool Call:[{"id": "some_unique_id", "type": "function", "function": {"name": "get_adjacent_relations_and_entities","arguments": {"entity": "wd:Q72792", "direction": "incoming"}}}]        
Suppose it returns:
property|propertyLabel|value|valueLabel
--|--|--|--
wdt:P162|producer|wd:Q375321|One Million B.C.
wdt:P162|producer|wd:Q3887452|Our Gang Follies of 1938
wdt:P162|producer|wd:Q7110646|Our Gang Follies of 1936
wdt:P162|producer|wd:Q3772025|
wdt:P162|producer|wd:Q16608539|
We see the property wdt:P162 (producer) that links to multiple entities, each with its own ID (e.g. wd:Q375321 (One Million B.C.), wd:Q3887452 (Our Gang Follies of 1938), wd:Q7110646 (Our Gang Follies of 1936)), wd:Q3772025, wd:Q16608539.
Final Answer: Hal Roach produced {5} films.
"""
]


requirement_prompt = """
When you want to call the tool:
    - Always follow the CORRECT format whenever you want to make a tool call.
    - Continue making tool calls until you arrive at a final textual answer. Then, and only then, stop making tool calls and provide your final answer in 'content'."""

requirement_prompt_short = """
When you want to call the tool:
    - Always follow the CORRECT format whenever you want to make a tool call."""

detailed_instruction = """
Furthermore,

    - Sometimes the 'get_adjacent_relations_and_entities' tool returns an entity ID ('value') without a corresponding entity name ('valueLabel'). If that occurs, continue making the correct tool calls using the entity ID ('value') alone, if necessary, until you find the information needed to answer the question. Relevant details may appear in subsequent tool calls.
    - Whenever get_adjacent_relations_and_entities returns multiple entities for a single relevant relation, you must examine every single one of those entities, even if there are tens or hundreds. Do not skip any; each could be relevant to the question.
    - If the question happens to be a 'when' question, you must provide the final answer with the value of the entity as given (i.e., in the format {1889-04-20} or {1889-04-20-08:00}) from the results of get_adjacent_relations_and_entities, as well as in the format {Month DD, YYYY}.
    - If the question asks for "how many" of something, you must provide the Arabic numeral (e.g. {7}), not the word (e.g. {seven}).
    - If searching from one direction does not yield information that seems relevant to the question, you may try searching from the other direction (e.g., "incoming" instead of "outgoing", or "outgoing" instead of "incoming") of the starting entity if you think it makes sense to try..
    - In your final answer, you must 1) write 'Final answer:' immediately before providing your final answer, 2) enclose the answer entity (or entities) in curly braces, and 3) use each entity name exactly as returned by the get_adjacent_relations_and_entities tool. For example, if the tool’s output shows "English Language", you must produce {English Language} (keeping the exact phrase) rather than shortening it to "English.".
    - If you cannot gather enough information using the tools to answer the question, rely on the information you already have, your reasoning abilities, and your own knowledge to produce the best possible answer(s)."""

detailed_instruction_qald = """
Furthermore,

    - Whenever get_adjacent_relations_and_entities returns multiple entities for a single relevant relation, you must examine every single one of those entities, even if there are tens or hundreds. Do not skip any; each could be relevant to the question.
    - If the question happens to be a boolean question (e.g., "Did ...", "Are ...", "Can ...", "Were ...", "Do ...", "Was ...", "Does", "Has ...", "Is ...",), you must include {true} or {false} in your final answer. Do not just say {yes} or {no}.
    - If the question happens to be a 'when' question, you must provide the final answer with the value of the entity as given (in the format {YYYY-MM-DDTHH:MM:SSZ}) from the results of get_adjacent_relations_and_entities, as well as in the format {Month DD, YYYY}.
    - If the question asks for "how many" of something, you must provide the Arabic numeral (e.g. {7}), not the word (e.g. {seven}).
    - If searching from one direction does not yield information that seems relevant to the question, you may try searching from the other direction (e.g., "incoming" instead of "outgoing", or "outgoing" instead of "incoming") of the starting entity if you think it makes sense to try.
    - In your final answer, you must 1) write 'Final answer:' immediately before providing your final answer, 2) enclose the answer entity (or entities) in curly braces, and 3) use each entity name exactly as returned by the get_adjacent_relations_and_entities tool. For example, if the tool’s output shows "English Language", you must produce {English Language} (keeping the exact phrase) rather than shortening it to "English.".
    - If you cannot gather enough information using the tools to answer the question, rely on the information you already have, your reasoning abilities, and your own knowledge to produce the best possible answer(s)."""

prompt = {
    "main": "You are a knowledgeable question-answering agent specializing in knowledge-graph question answering. You will receive a question and may call a tool to navigate the knowledge graph, collect information, and then formulate an answer.",
    "additional": {
        "CWQ": "Additionally, direction must be \"incoming\" or \"outgoing\"." + requirement_prompt + detailed_instruction,
        "WebQSP": "Additionally, direction must be \"incoming\" or \"outgoing\"." + requirement_prompt + detailed_instruction,
        "GrailQA": "Additionally, direction must be \"incoming\" or \"outgoing\"." + requirement_prompt + detailed_instruction,
        "SimpleQuestions": "Additionally, direction must be \"outgoing\"." + requirement_prompt_short + detailed_instruction,
        "QALD-9": "Additionally, direction must be \"incoming\" or \"outgoing\"." + requirement_prompt + detailed_instruction_qald,
        "QALD-10": "Additionally, direction must be \"incoming\" or \"outgoing\"." + requirement_prompt + detailed_instruction_qald,
    },
    "examples": {
        "CWQ": cwq_examples,
        "WebQSP": webqsp_examples,
        "GrailQA": grailqa_examples,
        "SimpleQuestions": simplequestions_examples,
        "QALD-9": qald_examples,
        "QALD-10": qald_examples,
    }
}

import random

def get_system_prompt(dataset, num_examples=0):
    assert dataset in ["SimpleQuestions", "WebQSP", "GrailQA", "QALD-9", "QALD-10", "CWQ"], f"Unsupported dataset: {dataset}"
    # IO mode is num_examples < 0
    if num_examples < 0:
        msg = "In your final answer, you must 1) write 'Final answer:' immediately before providing your final answer, 2) enclose the answer entity (or entities) in curly braces."
    else:
        msg = prompt["main"] + "" + prompt["additional"][dataset]
        msg += "\nBelow are examples of how to correctly make tool calls, interpret the tool call results, and produce the final answer(s).\n" + "\n\n".join(random.sample(prompt["examples"][dataset], k=num_examples-5)) + "\n--"
    return {"role": "system", "content": msg}


if __name__ == "__main__":
    print(get_system_prompt("WebQSP", num_examples=-1))
    
    print("=" * 80)

    print(get_system_prompt("CWQ", num_examples=2))

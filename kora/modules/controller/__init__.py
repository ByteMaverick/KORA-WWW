"""The controller's steps, and the one way the harness builds it."""


def build_controller(agent, llm, kg, arm, judge, budget, names=None, resolver=None,
                     property_choice="match", fallback="held"):
    from kora.agents.controller import Controller
    from kora.modules.controller.agent import TaskAgent
    from kora.modules.controller.strategies import strategy_for
    from kora.modules.planning.planner import Planner
    from kora.modules.reduction.property import Chooser
    planner = Planner(llm)
    strategy = strategy_for(arm, llm, kg)
    # `property_choice` is `KORA_PROPERTY`: "match" is the word match alone.
    chooser = None
    if property_choice == "model":
        chooser = Chooser(llm, kg)
    return Controller(TaskAgent(agent, kg), planner, judge, strategy,
                      kg=kg, budget=budget, names=names, resolver=resolver, chooser=chooser,
                      fallback=fallback)

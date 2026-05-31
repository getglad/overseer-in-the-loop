"""Red-team domain — drive adversarial actions past agent refusal into the gate.

Each attack is a proposed tool call with a known-correct verdict. The harness
feeds it straight to the action classifier (bypassing the agent's own refusal)
and asserts the gate decides as expected — proving the gate holds under pressure
rather than relying on the agent declining to misbehave.
"""

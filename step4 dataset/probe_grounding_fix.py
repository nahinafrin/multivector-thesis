import step_10_grounding_judge as s10

q1 = "Was Wilson president of the American Political Science Association in 1910 ?"
ev = ["Wilson was president of the American Political Science Association in 1910."]
print("positive lex:", s10.lexical_overlap("Yes.", ev, question=q1))
print("positive mdl:", s10.model_faithfulness("Yes.", ev, question=q1))

q2 = "Is Qatar bordered by Saudi Arabia to the south?"
print("negative lex:", s10.lexical_overlap("Yes.", ev, question=q2))
print("negative mdl:", s10.model_faithfulness("Yes.", ev, question=q2))

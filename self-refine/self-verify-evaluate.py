import json


filename = 'results/llama_rag_verify.json'
data = []
with open(filename) as file:
	for line in file.readlines():
		data.append(json.loads(line))
	# data = json.load(file)

acc = 0

for d in data:
	# print(type(d))
	correct_answer = float(d['correct_answer'])
	model_short_answer = float(d['model_short_answer'])
	acc += 1 if correct_answer == model_short_answer else 0

print(acc)

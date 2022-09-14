import dataset
import model
import time
import evaluation


# Evaluation

dataset = dataset.KorteRaw("KORTE")
network = model.get_yolo('checkpoints/pretrained.pt')
start = time.time_ns()
for i in range(0, 100, 5):
    print(f"Evaluating for confidence level \t {i/100}")
    print(evaluation.evaluate_box(network, dataset, i/100))

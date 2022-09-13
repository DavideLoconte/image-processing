import dataset
import model
import time
import evaluation


# Evaluation

dataset = dataset.KorteRaw("KORTE")
network = model.get_yolo('networks/yolov5x6.pt')
start = time.time_ns()
for i in range(0, 1, 0.05):
    print(f"Evaluating for threshold{i}")
    eval = evaluation.evaluate_box(network, dataset, i)
print(f"Eval in time {(time.time_ns() - start) / 1_000_000_000} s")
print(eval)


import os
import glob
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

base_dir = os.path.expanduser("/home/lmy/workspace/HARL/results/smac/3s5z/happo/happo-smac-3s5z/seed-00001-2025-05-17-14-33-44/logs/")
output_dir = os.path.expanduser("/home/lmy/workspace/HARL/results/merged_logs-happo-smac-3s5z-2025-05-17-14-33-44/")
os.makedirs(output_dir, exist_ok=True)

writers = {f"agent{i}": SummaryWriter(os.path.join(output_dir, f"agent{i}")) for i in range(8)}

for i in range(8):
    path = os.path.join(base_dir, f"agent{i}", "policy_loss", f"agent{i}","policy_loss")
    files = glob.glob(os.path.join(path, "events.out.tfevents.*"))
    if not files:
        continue
    acc = EventAccumulator(files[0])
    acc.Reload()
    #print(f"Available tags in agent{i}: {acc.Tags()['scalars']}")
    for event in acc.Scalars(f"agent{i}/policy_loss"):
        writers[f"agent{i}"].add_scalar("policy_loss", event.value, event.step)

for w in writers.values():
    w.close()
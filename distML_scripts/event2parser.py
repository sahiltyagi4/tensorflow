from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

eventfiles = '/Users/sahiltyagi/Desktop/dir26/eval/events.out.tfevents.1572414353.48af083a1369'
event_acc = EventAccumulator(eventfiles)
event_acc.Reload()
# Show all tags in the log file
print(event_acc.Tags())
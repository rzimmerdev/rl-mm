import time
from collections import defaultdict
from typing import List


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

# Benchmark class using Singleton
class Benchmark(metaclass=Singleton):
    def __init__(self):
        self.metrics = defaultdict(list)

    def start(self, metric_name: str):
        """Start timing a specific metric."""
        self.metrics[metric_name].append({"start": time.time(), "duration": None})

    def stop(self, metric_name: str):
        """Stop timing and calculate the duration for a specific metric."""
        if self.metrics[metric_name][-1]["duration"] is None:
            self.metrics[metric_name][-1]["duration"] = time.time() - self.metrics[metric_name][-1]["start"]

    def get(self, metric_name: str):
        """Get the total duration of a specific metric."""
        return sum(record["duration"] for record in self.metrics[metric_name] if record["duration"] is not None)

    def report(self):
        """Generate a report of all tracked metrics."""
        report = {}
        for metric, records in self.metrics.items():
            durations = [record["duration"] for record in records if record["duration"] is not None]
            report[metric] = {
                "count": len(durations),
                "mean_duration": sum(durations) / len(durations) if durations else 0,
                "total_duration": sum(durations),
            }
        return report

    @classmethod
    def report_cols(cls) -> List[str]:
        return ["count", "mean_duration", "total_duration"]

    def reset(self):
        """Clear all metrics."""
        self.metrics.clear()

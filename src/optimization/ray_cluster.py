import ray
import os

class RayClusterManager:
    """
    Phase 3.5: Distributed Compute Routing.
    Manages local workstation cores for parallel trial execution.
    """
    
    def __init__(self, reserve_cpus: int = 2):
        self.reserve_cpus = reserve_cpus

    def initialize_cluster(self) -> None:
        """
        Initializes Ray for distributed task management.
        Detects total CPUs and reserves specified cores for host OS.
        Detects GPUs if available.
        """
        if not ray.is_initialized():
            total_cpus = os.cpu_count() or 1
            num_cpus = max(1, total_cpus - self.reserve_cpus)
            
            # Simple GPU detection - in a real scenario, we might use nvidia-smi or similar
            # Ray handles most of this automatically if we don't specify, 
            # but we'll stick to the plan's explicit signature if needed.
            print(f"      - Initializing Ray with {num_cpus} CPUs (Total: {total_cpus}, Reserved: {self.reserve_cpus})")
            ray.init(num_cpus=num_cpus, ignore_reinit_error=True)

    def shutdown_cluster(self) -> None:
        """Gracefully shuts down the Ray cluster."""
        if ray.is_initialized():
            ray.shutdown()

#!/usr/bin/env python3
"""
Script de surveillance de la mémoire GPU pour détecter les fuites mémoire.

Usage:
    python monitor_gpu.py
"""

import time
import torch
from src.utils import get_gpu_memory_info, clear_gpu_memory
from rich.console import Console
from rich.table import Table
from rich.live import Live

console = Console()


def monitor_gpu_memory(interval=2.0, duration=None):
    """
    Surveille la mémoire GPU en temps réel.

    Args:
        interval (float): Intervalle entre les mesures en secondes
        duration (float): Durée totale de surveillance (None = infini)
    """
    console.rule("[bold blue]🔍 GPU Memory Monitor[/]")

    start_time = time.time()
    measurements = []

    def generate_table():
        table = Table(title="GPU Memory Usage")
        table.add_column("Time", justify="right")
        table.add_column("Used (MB)", justify="right")
        table.add_column("Free (MB)", justify="right")
        table.add_column("Total (MB)", justify="right")
        table.add_column("Usage %", justify="right")
        table.add_column("Trend", justify="center")

        for i, measurement in enumerate(measurements[-10:]):  # Dernières 10 mesures
            trend = ""
            if i > 0:
                prev_used = measurements[len(measurements) - 10 + i - 1]["used_mb"]
                current_used = measurement["used_mb"]
                if current_used > prev_used + 50:  # Plus de 50MB d'augmentation
                    trend = "📈"  # Augmentation
                elif current_used < prev_used - 50:  # Plus de 50MB de diminution
                    trend = "📉"  # Diminution
                else:
                    trend = "➡️"  # Stable

            elapsed = measurement["timestamp"] - start_time
            table.add_row(
                f"{elapsed:.1f}s",
                f"{measurement['used_mb']:.1f}",
                f"{measurement['free_mb']:.1f}",
                f"{measurement['total_mb']:.1f}",
                f"{measurement['usage_percent']:.1f}%",
                trend,
            )

        return table

    try:
        with Live(generate_table(), refresh_per_second=1) as live:
            while True:
                memory_info = get_gpu_memory_info()

                if "error" in memory_info:
                    console.print("❌ Error: CUDA not available")
                    break

                memory_info["timestamp"] = time.time()
                measurements.append(memory_info)

                # Détection de fuite mémoire
                if len(measurements) >= 5:
                    # Vérifier si la mémoire augmente constamment
                    recent_usage = [m["used_mb"] for m in measurements[-5:]]
                    if all(recent_usage[i] < recent_usage[i + 1] for i in range(4)):
                        # Mémoire augmente constamment
                        if recent_usage[-1] - recent_usage[0] > 200:  # Plus de 200MB
                            console.print(
                                "🚨 [bold red]MEMORY LEAK DETECTED![/] Memory increased by "
                                f"{recent_usage[-1] - recent_usage[0]:.1f}MB in last 5 measurements"
                            )

                live.update(generate_table())

                # Vérifier la durée
                if duration and (time.time() - start_time) >= duration:
                    break

                time.sleep(interval)

    except KeyboardInterrupt:
        console.print("\n⚠️ Monitoring stopped by user")

    # Statistiques finales
    if measurements:
        max_usage = max(m["used_mb"] for m in measurements)
        min_usage = min(m["used_mb"] for m in measurements)
        avg_usage = sum(m["used_mb"] for m in measurements) / len(measurements)

        console.rule("[bold green]📊 Final Statistics[/]")
        console.print(f"Max usage: {max_usage:.1f} MB")
        console.print(f"Min usage: {min_usage:.1f} MB")
        console.print(f"Average usage: {avg_usage:.1f} MB")
        console.print(f"Memory variation: {max_usage - min_usage:.1f} MB")

        if max_usage - min_usage > 500:  # Plus de 500MB de variation
            console.print(
                "🚨 [bold red]Large memory variation detected! Possible memory leak.[/]"
            )


def stress_test():
    """Test de stress pour détecter les fuites mémoire."""
    console.rule("[bold yellow]🧪 GPU Memory Stress Test[/]")

    initial_memory = get_gpu_memory_info()
    console.print(f"Initial memory usage: {initial_memory['used_mb']:.1f} MB")

    # Créer et supprimer des tensors plusieurs fois
    for i in range(10):
        console.print(f"Iteration {i + 1}/10...")

        # Créer des gros tensors
        tensors = []
        for j in range(5):
            if torch.cuda.is_available():
                tensor = torch.randn(1000, 1000, device="cuda")
                tensors.append(tensor)

        # Vérifier la mémoire après création
        after_creation = get_gpu_memory_info()
        console.print(f"  After creation: {after_creation['used_mb']:.1f} MB")

        # Supprimer les tensors
        del tensors

        # Forcer le nettoyage
        clear_gpu_memory()

        # Vérifier la mémoire après nettoyage
        after_cleanup = get_gpu_memory_info()
        console.print(f"  After cleanup: {after_cleanup['used_mb']:.1f} MB")

        time.sleep(1)

    final_memory = get_gpu_memory_info()
    console.print(f"Final memory usage: {final_memory['used_mb']:.1f} MB")

    memory_leak = final_memory["used_mb"] - initial_memory["used_mb"]
    if memory_leak > 100:  # Plus de 100MB de fuite
        console.print(f"🚨 [bold red]Memory leak detected: {memory_leak:.1f} MB![/]")
    else:
        console.print("[bold green]✅ No significant memory leak detected.[/]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPU Memory Monitor")
    parser.add_argument(
        "--mode",
        choices=["monitor", "stress"],
        default="monitor",
        help="Mode: monitor (surveillance continue) or stress (test de stress)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Intervalle entre les mesures (secondes)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Durée totale de surveillance (secondes, None = infini)",
    )

    args = parser.parse_args()

    if args.mode == "monitor":
        monitor_gpu_memory(args.interval, args.duration)
    else:
        stress_test()

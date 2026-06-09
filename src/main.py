"""Main entry point for the Adaptive Memory Ecology system."""

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.core.orchestrator import MemoryEcology
from src.core import MemoryType

console = Console()


async def interactive_session():
    """Run an interactive session with the memory ecology."""
    console.print(Panel.fit(
        "[bold green]Adaptive Memory Ecology[/bold green]\n"
        "[dim]Multi-layer biologically-inspired agentic memory system[/dim]",
        border_style="green",
    ))

    ecology = MemoryEcology()

    console.print("\n[bold]Commands:[/bold]")
    console.print("  [cyan]query <text>[/cyan]    — Query the memory system")
    console.print("  [cyan]add <text>[/cyan]      — Add a memory")
    console.print("  [cyan]consolidate[/cyan]     — Run consolidation cycle")
    console.print("  [cyan]health[/cyan]          — Show system health")
    console.print("  [cyan]feedback <y/n>[/cyan]  — Provide feedback on last response")
    console.print("  [cyan]quit[/cyan]            — Exit")
    console.print()

    while True:
        try:
            user_input = console.input("[bold blue]> [/bold blue]").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            break

        elif user_input.lower() == "health":
            health = ecology.get_system_health()
            table = Table(title="System Health")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            for key, value in health.items():
                if key == "meta_insights":
                    for insight in value:
                        table.add_row("insight", insight)
                elif key == "evolution_stats":
                    for k, v in value.items():
                        table.add_row(f"  evo_{k}", str(v))
                else:
                    table.add_row(key, str(value))
            console.print(table)

        elif user_input.lower() == "consolidate":
            with console.status("[bold yellow]Running consolidation cycle..."):
                stats = await ecology.consolidate()
            console.print(f"[green]✓ Consolidation complete:[/green] {stats}")

        elif user_input.lower().startswith("feedback"):
            parts = user_input.split()
            if len(parts) > 1 and parts[1].lower() in ("y", "yes"):
                ecology.provide_feedback(True)
                console.print("[green]✓ Positive feedback recorded[/green]")
            else:
                ecology.provide_feedback(False)
                console.print("[yellow]✓ Negative feedback recorded[/yellow]")

        elif user_input.lower().startswith("add "):
            content = user_input[4:].strip()
            with console.status("[bold yellow]Storing memory..."):
                memory_id = await ecology.add_memory(content)
            console.print(f"[green]✓ Memory stored:[/green] {memory_id[:8]}...")

        elif user_input.lower().startswith("query "):
            query = user_input[6:].strip()
            with console.status("[bold yellow]Querying memory ecology..."):
                response = await ecology.query(query)
            console.print(Panel(response, title="Response", border_style="green"))

        else:
            # Default: treat as query
            with console.status("[bold yellow]Querying memory ecology..."):
                response = await ecology.query(user_input)
            console.print(Panel(response, title="Response", border_style="green"))


def main():
    """Entry point."""
    asyncio.run(interactive_session())


if __name__ == "__main__":
    main()

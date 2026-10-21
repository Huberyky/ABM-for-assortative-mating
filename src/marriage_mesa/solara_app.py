"""Solara entry point used by ``marriage-mesa visualize``."""

from marriage_mesa.visualization import dashboard_from_environment

page = dashboard_from_environment()

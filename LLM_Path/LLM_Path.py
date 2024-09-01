"""Welcome to Reflex! This file outlines the steps to create a basic app."""

import reflex as rx
import plotly as plt

from rxconfig import config


class State(rx.State):
    """The app state."""

    ...


def index() -> rx.Component:
    # Welcome Page (Index)
    return rx.container(
        rx.text("Welcome to LLM Path"),
        rx.plotly(

        )
    )


app = rx.App()
app.add_page(index)

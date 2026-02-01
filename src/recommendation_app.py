from nicegui import ui
import sqlite3
import pandas as pd

from paths import db_path

class Page:
    def __init__(self):
        self.column = ui.column().classes('self-center width-full')
        self.set_df: pd.DataFrame = self.get_predicted_scores()
        self.current_set = None
        self.current_score = 0
        self.sample_set()
        self.build_ui()


    def get_predicted_scores(self) -> pd.DataFrame:
        with sqlite3.connect(db_path()) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predicted_scores")
            rows = cursor.fetchall()

        df = pd.DataFrame(rows, columns=["id", "a", "b", "c", "rating"]).drop(columns=["id"]).dropna()
        df["w"] = (df["rating"] + 1) ** 5
        return df

    def get_dish_data(self, name):
        with sqlite3.connect(db_path()) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT raskas, hinta FROM dishes WHERE nimi==?", [name])
            row = cursor.fetchone()
            return row

    def create_card(self, name):
        raskas, hinta = self.get_dish_data(name)
        with ui.card().style("width: 100%; max-width: 1000px"), ui.row().classes('w-full items-center justify-between'):
            ui.label(name)
            with ui.column().style('align-items: flex-end'):
                with ui.row().style("gap: 1px"):
                    for i in range(hinta):
                        ui.icon("euro")
                with ui.row().style("gap: 1px"):
                    for i in range(raskas):
                        ui.icon("water_drop")

    def sample_set(self):
        result = self.set_df.sample(n=1, weights="w").values[0]
        self.current_set = result[:3]
        self.current_score = round(result[3], 1)

    def click_handler(self):
        self.sample_set()
        self.build_ui()

    def build_ui(self, skeleton=False):
        self.column.clear()
        with self.column.style('align-items: center'):
            ui.html("<div>", sanitize=False).style("height: 2em")
            if skeleton or not self.current_set:
                ui.skeleton(height="5em", width="15em")
                ui.skeleton(height="5em", width="15em")
                ui.skeleton(height="5em", width="15em")
            else:
                self.create_card(self.current_set[0])
                self.create_card(self.current_set[1])
                self.create_card(self.current_set[2])
                ui.label(f"Arvosana: {self.current_score}/5")
            
            ui.html("<div>", sanitize=False).style("height: 2em")
            ui.button("SEURAAVA", on_click=self.click_handler)
            
    ###

page = Page()

ui.run(title="Ruoka-suosittelu", host="0.0.0.0", port=8080)
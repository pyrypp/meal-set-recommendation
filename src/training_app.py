from nicegui import ui
import sqlite3
import asyncio

from paths import db_path

class Page:
    def __init__(self):
        self.current_id = None
        self.current_set = self.get_set()

        self.column = ui.column().classes('self-center width-full')
        self.build_ui()


    def get_set(self):
        with sqlite3.connect(db_path()) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, a, b, c FROM training WHERE rating IS NULL")
            row = cursor.fetchone()
            self.current_id = row[0]
            return row[1:]

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

    def set_rating(self, set_id, value):
        with sqlite3.connect(db_path()) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE training SET rating = ? WHERE id = ?", (value, set_id))
            conn.commit()

    async def rating_handler(self, event):
        await asyncio.sleep(0.2)
        self.build_ui(skeleton=True)
        await asyncio.sleep(0.1)
    
        self.set_rating(self.current_id, event.value)
        self.current_set = self.get_set()

        await asyncio.sleep(1)
        self.build_ui()
        await asyncio.sleep(0.1)

    def build_ui(self, skeleton=False):
        self.column.clear()
        with self.column.style('align-items: center'):
            ui.label(str(self.current_id))
            ui.html("<div>", sanitize=False).style("height: 2em")

            if skeleton:
                ui.skeleton(height="5em", width="15em")
                ui.skeleton(height="5em", width="15em")
                ui.skeleton(height="5em", width="15em")
            else:
                self.create_card(self.current_set[0])
                self.create_card(self.current_set[1])
                self.create_card(self.current_set[2])
            
            ui.html("<div>", sanitize=False).style("height: 2em")
            ui.rating(size='3em', on_change=self.rating_handler)
            
    ###

page = Page()

ui.run(title="Ruoka-arviointi", host="0.0.0.0", port=8080)
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tkinter import Tk, Checkbutton, IntVar, Label, Button

# Database connection settings
DB_PARAMS = {
    "dbname": "labmate",
    "user": "lm_u_a6sd78as7d6f78",
    "password": "8730fehoypd9ugcoa(&#*OuDId7&AT*WP8p9yp&W*DU&Gsd;oij;coduwe;yiouwdhfe",
    "host": "labmate.v2024.pl",
    "port": "5432"
}

# Global variables to store filters
selected_antennas = {}
selected_epcs = {}

# Initialize plot
fig, ax = plt.subplots()

def get_db_data():
    """Fetch data from PostgreSQL+TimescaleDB."""
    conn = psycopg2.connect(**DB_PARAMS)
    query = "SELECT time, rx_signal_strength, antno, inventory_epc FROM invm ORDER BY time DESC LIMIT 10"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def update_plot(frame):
    """Query DB and update the plot in real-time."""
    df = get_db_data()

    ax.clear()
    ax.set_title("Real-time RX Signal Strength")
    ax.set_xlabel("Time")
    ax.set_ylabel("RX Signal Strength")

    # Filter based on selected antno and epc
    for (ant, epc), checked in selected_antennas.items():
        if checked.get():
            filtered_df = df[(df['antno'] == ant) & (df['epc'] == epc)]
            ax.plot(filtered_df["time"], filtered_df["rx_signal_strength"], label=f"Ant {ant}, EPC {epc}")

    ax.legend(loc="upper right")

def create_gui():
    """Creates a Tkinter GUI for filtering options."""
    root = Tk()
    root.title("Select Filters")

    # Fetch initial data to get unique antennas and EPCs
    df = get_db_data()
    unique_filters = df.groupby(["antno", "epc"]).size().reset_index()[["antno", "epc"]]

    Label(root, text="Select Antennas and EPCs to Plot:").pack()

    for _, row in unique_filters.iterrows():
        ant, epc = row["antno"], row["epc"]
        var = IntVar(value=1)  # Default checked
        Checkbutton(root, text=f"Ant {ant}, EPC {epc}", variable=var).pack()
        selected_antennas[(ant, epc)] = var

    Button(root, text="Update Plot", command=lambda: None).pack()

    root.mainloop()

# Start real-time plot animation
ani = animation.FuncAnimation(fig, update_plot, interval=2000)  # Refresh every 2 sec

# Run GUI in a separate thread
import threading
threading.Thread(target=create_gui, daemon=True).start()

plt.show()
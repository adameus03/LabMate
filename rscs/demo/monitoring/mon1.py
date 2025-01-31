import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tkinter import Tk, Checkbutton, Button, Label, StringVar, IntVar

# Database connection parameters
DB_NAME = ''
DB_USER = ''
DB_PASSWORD = ''
DB_HOST = ''
DB_PORT = ''

def load_env():
    global DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT
    with open('.env') as f:
        for line in f:
            key, value = line.strip().split('=')
            if key == 'DB_NAME':
                DB_NAME = value
            elif key == 'DB_USER':
                DB_USER = value
            elif key == 'DB_PASSWORD':
                DB_PASSWORD = value
            elif key == 'DB_HOST':
                DB_HOST = value
            elif key == 'DB_PORT':
                DB_PORT = value


is_db_connected = False
should_disconnect_db = False
db_conn = None
# Function to query the database
def query_database(antno, epc):
    if not is_db_connected:
      db_conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
    query = f"""
    SELECT time, rx_signal_strength
    FROM invm
    WHERE antno = {antno} AND inventory_epc = '{epc}'
    ORDER BY time DESC
    LIMIT 100;
    """
    df = pd.read_sql(query, db_conn)
    if should_disconnect_db:
      db_conn.close()
    return df

# Function to update the plot
def update_plot(frame):
    plt.cla()  # Clear the current axes
    for antno, epc, var in filters:
        if var.get() == 1:  # Check if the checkbox is selected
            df = query_database(antno, epc)
            #df = df[df['rx_signal_strength'] >= 175] # Improve readability (skip zero values while still showing that the signal was lost)
            plt.plot(df['time'], df['rx_signal_strength'], label=f'antno={antno}, epc={epc}')
    plt.legend(loc='upper left')
    plt.xlabel('Time')
    plt.ylabel('RX Signal Strength')
    plt.ylim(175, None)  # Set the y-axis limit to 175 and the maximum value
    plt.title('Real-Time RX Signal Strength')
    plt.gcf().autofmt_xdate()

load_env()

# GUI setup
root = Tk()
root.title("Real-Time RX Signal Strength Plotter")

# Example filters (antno, epc, checkbox variable)
filters = [
    (0, 'aaaaaaaaaaaaaaaaaaaaaaaa', IntVar()),
    (1, 'aaaaaaaaaaaaaaaaaaaaaaaa', IntVar()),
    (0, 'bbbbbbbbbbbbbbbbbbbbbbbb', IntVar()),
    (1, 'bbbbbbbbbbbbbbbbbbbbbbbb', IntVar()),
    (0, 'cccccccccccccccccccccccc', IntVar()),
    (1, 'cccccccccccccccccccccccc', IntVar()),
    (0, 'dddddddddddddddddddddddd', IntVar()),
    (1, 'dddddddddddddddddddddddd', IntVar()),
    (0, 'eeeeeeeeeeeeeeeeeeeeeeee', IntVar()),
    (1, 'eeeeeeeeeeeeeeeeeeeeeeee', IntVar())
]

# Create checkboxes for each filter
for i, (antno, epc, var) in enumerate(filters):
    Checkbutton(root, text=f'antno={antno}, epc={epc}', variable=var).grid(row=i, column=0, sticky='w')

# Start animation
ani = animation.FuncAnimation(plt.gcf(), update_plot, interval=500)

# Show the plot
plt.show()

# Start the GUI event loop
root.mainloop()

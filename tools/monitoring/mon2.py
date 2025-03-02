import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

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
def query_database_all():
    if not is_db_connected:
      db_conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
    query = f"""
    SELECT time, rx_signal_strength, antno, inventory_epc
    FROM invm
    ORDER BY time DESC
    LIMIT 100
    """
    df = pd.read_sql(query, db_conn)
    if should_disconnect_db:
      db_conn.close()
    return df

def random_color():
    return random.choice(plt.cm.viridis.colors)

epc_color_map = {}

def epc_get_color(epc):
    global epc_color_map
    if epc not in epc_color_map:
        epc_color_map[epc] = random_color()
    return epc_color_map[epc]

# Function to update the plot
def update_plot(frame):
    plt.cla()  # Clear the current axes
    # for antno, epc in filters:
    #     df = query_database(antno, epc)
    #     #df = df[df['rx_signal_strength'] >= 175] # Improve readability (skip zero values while still showing that the signal was lost)
    #     plt.plot(df['time'], df['rx_signal_strength'], label=f'antno={antno}, epc={epc}', marker='o', linestyle='dashed')
    df = query_database_all()
    distinct_epcs = df['inventory_epc'].unique()
    # for antno in range(2):
    #     for epc in distinct_epcs:
    #         df_ant_epc = df[(df['antno'] == antno) & (df['inventory_epc'] == epc)]
    #         #time-domain plot
    #         #plt.plot(df_ant_epc['time'], df_ant_epc['rx_signal_strength'], label=f'antno={antno}, epc={epc}', marker='o', linestyle='dashed')
    #         #[rssi1, rssi2] parametric plot
    #         plt.plot(df_ant_epc['rx_signal_strength'], df_ant_epc['rx_signal_strength'], label=f'antno={antno}, epc={epc}', marker='o', linestyle='dashed')

    distinct_antnos = df['antno'].unique()
    num_antennas = len(distinct_antnos)
    
    # set cycle_ix as row index/num_antennas
    df['cycle_ix'] = df.index // num_antennas
    # combine rows sharing the same cycle_ix into one row with rssi values for different antennas crushed into an array
    df = df.groupby(['cycle_ix', 'inventory_epc'])['rx_signal_strength'].apply(list).reset_index()
    # rename rssi_signal_strength column to rssi
    df = df.rename(columns={'rx_signal_strength': 'rssi'})
    # filter out rssi vectors containing a 0 in any of the antennas
    #df = df[df['rssi'].apply(lambda x: all([rssi > 0 for rssi in x]))]
    # change rssi (range 180-200 or 0) to range 0-1
    df['rssi'] = df['rssi'].apply(lambda x: [0 if rssi == 0 else (rssi - 180) / 20 for rssi in x])

    
    # plot (rssi[0], rssi[1]) versus time - for each epc in different color
    for i, epc in enumerate(distinct_epcs):
        color = epc_get_color(epc)
        df_epc = df[df['inventory_epc'] == epc]
        # write the plot on the graph too
        plt.plot(df_epc['rssi'].apply(lambda x: x[0]), df_epc['rssi'].apply(lambda x: x[1]), label=f'epc={epc}', marker='o', linestyle='dashed', color=color, alpha=0.5, linewidth=1, markersize=5, markerfacecolor=color)
        # write the epc on the graph
        if len(df_epc) > 0:
            plt.text(df_epc['rssi'].apply(lambda x: x[0]).iloc[-1], df_epc['rssi'].apply(lambda x: x[1]).iloc[-1], epc, fontsize=9, color=color)
        else:
            print(f'epc={epc} has no data')
            #exit()
        
            
    # legend to the side
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    #plt.xlabel('Time')
    #plt.ylabel('RX Signal Strength')
    plt.xlabel('RSSI[0]')
    plt.ylabel('RSSI[1]')
    #plt.ylim(175, None)  # Set the y-axis limit to 175 and the maximum value
    plt.title('Real-Time RX Signal Strength')
    plt.gcf().autofmt_xdate()

load_env()

# Start animation
ani = animation.FuncAnimation(plt.gcf(), update_plot, interval=500)

# Show the plot
plt.show()


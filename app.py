import streamlit as st
import requests
import pytz
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from cycler import cycler
import seaborn as sns
from datetime import datetime, timedelta
 

# Load your data with st.cache_data
@st.cache_data

def load_data():
    # Replace this with your actual dataset
    df_st = pd.DataFrame({
        'date': pd.date_range(start='2024-12-28', periods=1, freq='D'),
    })
    return df_st

# Load data
data = load_data()
# Sidebar for input
st.sidebar.header("Obdobje validacije")
# Select start and end dates
start_date = st.sidebar.date_input("Začetni dan", value=data['date'].min())
end_date = st.sidebar.date_input("Končni dan", value=data['date'].max())

if start_date > end_date:
    st.sidebar.error("Začetni datum mora biti enak ali večji končnemu.")

# Prepare datetime objects
start_time = datetime.combine(start_date, datetime.min.time())
end_time = datetime.combine(end_date, datetime.max.time())
tab1, tab2 = st.tabs(["Informacije o CNE kršitvah", "Pretoki in neto pozicije"])
## --------------------------------------------------------------------------- ZAVIHEK 1 ----------------------------------------------------------------------------------------
with tab1:
    # Ensure a single button instance
    if "Pridobi podatke" not in st.session_state:
        st.session_state.button_clicked = False

    if st.button("Pridobi podatke", key="apply_filter_1"):
        st.session_state.button_clicked = True

    st.header("Informacije o CNE kršitvah")
    # Display the selected date range
    st.write("Izbrano časovno obdobje:")
    st.write(f"Začetek: {start_time}")
    st.write(f"Konec: {end_time}")

    # Vhodni podatki
    url = "https://publicationtool.jao.eu/core/api/data/finalComputation"
    base_params = {
        "Filter": '{"Tso":["10XSI-ELES-----1"]}',
        "Skip": 0,
        "Take": 100
    }

    all_data = []
    # Button to trigger the API request
    if st.session_state.button_clicked:
        # Make sure valid dates are selected
        if start_date <= end_date:
            # Fetch data from the API
            current_time = start_time
            while current_time <= end_time:
                from_utc = current_time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
                to_utc = (current_time + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
                
                params = base_params.copy()
                params["FromUtc"] = from_utc
                params["ToUtc"] = to_utc

                try:
                    response = requests.get(url, params=params, verify=False)
                    response.raise_for_status()
                    data = response.json()
                    if 'data' in data:
                        all_data.extend(data['data'])
                    else:
                        print(f"No 'data' key found in response for {from_utc} to {to_utc}.")
                except requests.exceptions.RequestException as e:
                    print(f"An error occurred for {from_utc} to {to_utc}: {e}")

                current_time += timedelta(hours=1)

            if all_data:
                df_origin = pd.DataFrame(all_data)
            else:
                print("No data was retrieved.") 
            if response.status_code == 200:
                # Parse the response
                api_data = response.json()
                st.success("Podatki uspešno pridobljeni!")
            else:
                st.error(f"Napaka pri pridobivanju podatkov: {response.status_code}")
        else:
            st.error("Izberni ustrezni datum za pridobivanje podatkov.")

        df = df_origin.copy()
        df = df.drop(['id','tso', 'cneEic','hubFrom','hubTo','substationFrom','substationTo', 'contingencies','elementType','fmaxType','cneStatus', 'presolved','imax','u','justification','contTso','ltaMargin','minRamFactor'], axis=1)
        df = df.iloc[:, :-14]

        # Convert to Ljubljana time zone (Europe/Ljubljana)
        df['dateTimeUtc'] = pd.to_datetime(df['dateTimeUtc'], utc=True)
        ljubljana_tz = pytz.timezone('Europe/Ljubljana')
        df['dateTimeUtc'] = df['dateTimeUtc'].dt.tz_convert(ljubljana_tz)
        df['dateTimeUtc'] = df['dateTimeUtc'].dt.tz_localize(None)

        df['contName'] = df['contName'].astype(str)
        df['contName'] = df['contName'].apply(lambda x: ' '.join(x.split()[4:]) if len(x.split()) > 6 else x)
        df['MACZT'] = (df['fmax'] - df['frm'] - df['fall'] + df['amr'] - df['iva'])/df['fmax']*100
        df['RAM_calc'] = df['fmax'] - df['frm'] - df['fall'] + df['amr'] - df['iva']

        df['contName'] = df['contName'].replace('None', 'BASECASE')
        df['direction'] = df['direction'].apply(lambda x: f"({x})")
        df['contName'] = df['contName'].replace('kV Cirkovce - Zerjavinec', '400 kV Cirkovce - Zerjavinec')
        df['contName'] = df['contName'].replace('Divaca - Pehlin', '220 kV Divaca - Pehlin')
        df['contName'] = df['contName'].replace('Bericevo-Divaca', '400 kV Bericevo - Divaca')
        df['cneName'] = df['cneName'].replace('Cirkovce - Heviz', '400 kV Cirkovce - Heviz')

        df['contName'] = df['contName'].apply(lambda x: f"(N-1) {x}" if x != 'BASECASE' else x)
        df['info'] = df['cneName'] + ' ' + df['direction'] + ' / ' + df['contName']
        df = df.drop(['cneName','direction','contName'], axis=1)
        df['info'] = df['info'].apply(lambda x: x.replace(" / ", "\n") if " / " in x else x)

        # df = df[df['iva'] > 0]
        # df = df[df['amr'] > 0]
        df = df[df['MACZT'] < 70]

        df['Hour_Minute'] = df['dateTimeUtc'].dt.strftime('%H:%M')

        # Main app layout
        st.write("Tabela elementov")
        st.dataframe(df, width=1000, height=400)

        # Define font sizes in one place
        font_sizes = {
            "title": 18,
            "labels": 16,
            "ticks": 10,
            "annot": 14  # For annotations in the heatmap
        }
        # ********************************** PLOT 1 *************************************
        st.write("MACZT po MTU za CNE s kršitvijo")
        heatmap_data = df.pivot_table(
            columns='info',
            index='Hour_Minute',
            values='MACZT',  # The data to be visualized (can be counts or another metric)
            aggfunc='sum'  # Use 'sum', 'mean', etc., depending on your needs
        )

        fig_1 = plt.figure(figsize=(20, 8))
        sns.heatmap(
            heatmap_data,
            cmap="RdYlGn",
            annot=True,
            fmt=".1f",
            linewidths=2.5,
            linecolor='black',
            cbar_kws={'label': 'MACZT [%]'},
            annot_kws={'size': font_sizes["annot"]}  # Font size for annotations
        )
        # plt.title("MACZT po MTU za CNE s kršitvijo", fontsize=font_sizes["title"])
        plt.xlabel("CNE in kršitev", fontsize=font_sizes["labels"])
        plt.ylabel("MTU", fontsize=font_sizes["labels"])
        plt.xticks(rotation=0, fontsize=font_sizes["ticks"])
        plt.yticks(rotation=0, fontsize=font_sizes["ticks"])
        plt.tight_layout()
        plt.show()

        # Show the plot in Streamlit
        st.pyplot(fig_1)

        # ********************************** PLOT 2 *************************************
        heatmap_data = df.pivot_table(
            columns='info',
            index='Hour_Minute',
            values='iva',  # The data to be visualized (can be counts or another metric)
            aggfunc='sum'  # Use 'sum', 'mean', etc., depending on your needs
        )

        st.write("IVA po MTU za CNE s kršitvijo")
        fig_2 = plt.figure(figsize=(20, 8))
        sns.heatmap(
            heatmap_data,
            cmap="RdYlGn_r",
            annot=True,
            fmt=".1f",
            linewidths=2.5,
            linecolor='black',
            cbar_kws={'label': 'IVA [%]'},
            annot_kws={'size': font_sizes["annot"]}  # Font size for annotations
        )
        # plt.title("MACZT po MTU za CNE s kršitvijo", fontsize=font_sizes["title"])
        plt.xlabel("CNE in kršitev", fontsize=font_sizes["labels"])
        plt.ylabel("MTU", fontsize=font_sizes["labels"])
        plt.xticks(rotation=0, fontsize=font_sizes["ticks"])
        plt.yticks(rotation=0, fontsize=font_sizes["ticks"])
        plt.tight_layout()
        plt.show()

        # Show the plot in Streamlit
        st.pyplot(fig_2)

##--------------------------------------------------------------------------- ZAVIHEK 2 ----------------------------------------------------------------------------------------
with tab2:
    st.header("Informacija o pretokih na mejah držav")
    # Display the selected date range
    st.write("Izbrano časovno obdobje:")
    st.write(f"Začetek: {start_time}")
    st.write(f"Konec: {end_time}")

    # Ensure a single button instance
    if "Pridobi podatke" not in st.session_state:
        st.session_state.button_clicked = False

    if st.button("Pridobi podatke", key="apply_filter"):
        st.session_state.button_clicked = True

    # ********************************** PLOT 1 *************************************
    # Vhodni podatki
    url = "https://publicationtool.jao.eu/core/api/data/refprog"

    all_data = []
    # Button to trigger the API request
    # if st.sidebar.button("Pridobi podatke 2"):
    if st.session_state.button_clicked:
        # Make sure valid dates are selected
        if start_date <= end_date:
            # Fetch data from the API
            current_time = start_time
            from_utc = current_time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            to_utc = (current_time + timedelta(hours=23)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
            params_2 = {
            "FromUtc": from_utc,
            "ToUtc": to_utc
            }
            try:
                response = requests.get(url, params=params_2, verify=False)
                response.raise_for_status()
                data = response.json()
                if 'data' in data:
                    all_data.extend(data['data'])
                else:
                    print(f"No 'data' key found in response for {from_utc} to {to_utc}.")
            except requests.exceptions.RequestException as e:
                print(f"An error occurred for {from_utc} to {to_utc}: {e}")

            if all_data:
                df = pd.DataFrame(all_data)
            else:
                print("No data was retrieved.") 
            if response.status_code == 200:
                # Parse the response
                api_data = response.json()
                st.success("Podatki uspešno pridobljeni!")
            else:
                st.error(f"Napaka pri pridobivanju podatkov: {response.status_code}")
        else:
            st.error("Izberni ustrezni datum za pridobivanje podatkov.")
        # Extract the 'data' list
        data_list = data['data']

        # Create a DataFrame
        df = pd.DataFrame(data_list)
        df['dateTimeUtc'] = pd.to_datetime(df['dateTimeUtc'], utc=True)
        # Convert to Ljubljana time zone (Europe/Ljubljana)
        ljubljana_tz = pytz.timezone('Europe/Ljubljana')
        df['dateTimeUtc'] = df['dateTimeUtc'].dt.tz_convert(ljubljana_tz)
        df['dateTimeUtc'] = df['dateTimeUtc'].dt.tz_localize(None)

        # Filter columns starting with 'border_SI_'
        filtered_columns = [col for col in df.columns if 'SI' in col]

        # Create a new DataFrame with only the filtered columns
        df_slo = df[filtered_columns]
        # Dictionary to map country codes to country names
        country_mapping = {
            'AT': 'Avstrija',
            'BE': 'Belgija',
            'CZ': 'Češka',
            'DE': 'Nemčija',
            'FR': 'Francija',
            'HR': 'Hrvaška',
            'HU': 'Madžarska',
            'NL': 'Nizozemska',
            'PL': 'Poljska',
            'RO': 'Romunija',
            'SI': 'Slovenija',
            'SK': 'Slovaška',
            'IT': 'Italija'
        }

        # Function to map country code to full name
        def map_country_name(col_name):
            # Extract the country code (last two characters)
            country_code = col_name.split('_')[1]
            # Map the country code to the full country name
            country_name = country_mapping.get(country_code, country_code)  # Default to country code if not found
            return f"{country_name}"

        # Change column names by applying the mapping
        df_slo.columns = [map_country_name(col) if 'border_' in col else col for col in df_slo.columns]
        df_slo = df_slo.round(0).astype(int)


        date_str = df['dateTimeUtc'].iloc[0].strftime('%d.%m.%Y')
        # Set the plot title with the actual date
        title = f'Izračunane izmenjave pretokov na mejnih povezavah Slovenije {date_str}'

        # Create a custom color cycle
        color_cycle = cycler('color', plt.cm.tab10.colors)  # Using the tab20 color map

        # Apply the color cycle to the current plot
        plt.rcParams['axes.prop_cycle'] = color_cycle

        # Plotting all columns in the df_max DataFrame
        fig_3 = plt.figure(figsize=(12, 6))

        # Set the line width to 2 (or any value you prefer)
        line_width = 3  # You can increase this value to make the lines thicker

        for column in df_slo.columns:
            plt.plot(df['dateTimeUtc'], df_slo[column], label=column, linewidth=line_width, zorder=3)  # Set higher zorder for lines

        # Set the plot title dynamically
        plt.title(title)

        # Set the x-axis and y-axis labels
        plt.xlabel('Čas (HH:MM)')
        plt.ylabel('Izmenjane količine [MW]')

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=0)

        # Format x-axis to show only hours and minutes (HH:MM)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        # Enable grid and set a lower zorder to place it behind the lines
        plt.grid(True, zorder=0)  # Set lower zorder for grid

        # Change x-axis tick size
        plt.tick_params(axis='x', labelsize=12)  # Adjust labelsize as needed (e.g., 12)

        # Get the current handles and labels from the plot
        handles, labels = plt.gca().get_legend_handles_labels()

        # Create custom handles with thicker lines for the legend
        custom_handles = [Line2D([0], [0], color=handle.get_color(), lw=4, solid_capstyle='round') for handle in handles]

        # Display the legend with custom handles and labels
        plt.legend(
            handles=custom_handles,
            labels=labels,
            fontsize=10,        # Optional: Adjust the font size for the legend labels
            loc='upper center',
            bbox_to_anchor=(0.5, -0.1),
            ncol=6  # You can change this number to adjust the number of columns in the legend
        )
        st.pyplot(fig_3)
    
    # ********************************** PLOT 2 *************************************
    url = "https://publicationtool.jao.eu/core/api/data/maxNetPos"
    all_data = []
    # Button to trigger the API request
    # if st.sidebar.button("Pridobi podatke 2"):
    if st.session_state.button_clicked:
        # Make sure valid dates are selected
        if start_date <= end_date:
            # Fetch data from the API
            current_time = start_time
            from_utc = current_time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            to_utc = (current_time + timedelta(hours=23)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
            params_2 = {
            "FromUtc": from_utc,
            "ToUtc": to_utc
            }
            try:
                response = requests.get(url, params=params_2, verify=False)
                response.raise_for_status()
                data = response.json()
                if 'data' in data:
                    all_data.extend(data['data'])
                else:
                    print(f"No 'data' key found in response for {from_utc} to {to_utc}.")
            except requests.exceptions.RequestException as e:
                print(f"An error occurred for {from_utc} to {to_utc}: {e}")

            if all_data:
                df = pd.DataFrame(all_data)
            else:
                print("No data was retrieved.") 
            if response.status_code == 200:
                # Parse the response
                api_data = response.json()
                st.success("Podatki uspešno pridobljeni!")
            else:
                st.error(f"Napaka pri pridobivanju podatkov: {response.status_code}")
        else:
            st.error("Izberni ustrezni datum za pridobivanje podatkov.")
        
        # Extract the 'data' list
        data_list = data['data']
        # Create a DataFrame
        df = pd.DataFrame(data_list)
        df['dateTimeUtc'] = pd.to_datetime(df['dateTimeUtc'], utc=True)

        # Convert to Ljubljana time zone (Europe/Ljubljana)
        ljubljana_tz = pytz.timezone('Europe/Ljubljana')
        df['dateTimeUtc'] = df['dateTimeUtc'].dt.tz_convert(ljubljana_tz)
        df['dateTimeUtc'] = df['dateTimeUtc'].dt.tz_localize(None)
        df = df.drop(columns=['minALBE', 'minALDE', 'maxALBE', 'maxALDE'])

        # Create the DataFrame for min values
        df_min = df.filter(regex='^min', axis=1)
        # Create the DataFrame for max values
        df_max = df.filter(regex='^max', axis=1)

        # Mapping of old column names to Slovenian names for 'min' columns
        min_column_mapping = {
            'minAT': 'Avstrija',
            'minBE': 'Belgija',
            'minCZ': 'Češka',
            'minDE': 'Nemčija',
            'minHR': 'Hrvaška',
            'minHU': 'Madžarska',
            'minFR': 'Francija',
            'minPL': 'Poljska',
            'minSI': 'Slovenija',
            'minSK': 'Slovaška',
            'minRO': 'Romunija',
            'minNL': 'Nizozemska',
            # Add more mappings as needed
        }

        # Mapping of old column names to Slovenian names for 'max' columns
        max_column_mapping = {
            'maxAT': 'Avstrija',
            'maxBE': 'Belgija',
            'maxCZ': 'Češka',
            'maxDE': 'Nemčija',
            'maxHR': 'Hrvaška',
            'maxHU': 'Madžarska',
            'maxFR': 'Francija',
            'maxPL': 'Poljska',
            'maxSI': 'Slovenija',
            'maxSK': 'Slovaška',
            'maxRO': 'Romunija',
            'maxNL': 'Nizozemska'
            # Add more mappings as needed
        }

        # Rename columns for the 'min' DataFrame
        df_min = df_min.rename(columns=min_column_mapping)
        # Rename columns for the 'max' DataFrame
        df_max = df_max.rename(columns=max_column_mapping)

        # Extract the date from the first entry in 'dateTimeUtc' column and format it
        df['dateTimeUtc'] = pd.to_datetime(df['dateTimeUtc'])
        date_str = df['dateTimeUtc'].iloc[0].strftime('%d.%m.%Y')

        # Set the plot title with the actual date
        title = f'Maksimalne neto pozicije držav za dan {date_str}'

        # Create a custom color cycle
        color_cycle = cycler('color', plt.cm.tab20.colors)  # Using the tab20 color map

        # Apply the color cycle to the current plot
        plt.rcParams['axes.prop_cycle'] = color_cycle

        # Plotting all columns in the df_max DataFrame
        fig_4 = plt.figure(figsize=(12, 6))

        line_width = 3  # You can increase this value to make the lines thicker

        for column in df_max.columns:
            plt.plot(df['dateTimeUtc'], df_max[column], label=column, linewidth=line_width, zorder=3)  # Set higher zorder for lines

        # Set the plot title dynamically
        plt.title(title)

        # Set the x-axis and y-axis labels
        plt.xlabel('Čas (HH:MM)')
        plt.ylabel('Neto pozicija [MW]')

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=0)

        # Format x-axis to show only hours and minutes (HH:MM)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        # Enable grid and set a lower zorder to place it behind the lines
        plt.grid(True, zorder=0)  # Set lower zorder for grid

        # Change x-axis tick size
        plt.tick_params(axis='x', labelsize=12)  # Adjust labelsize as needed (e.g., 12)

        # Get the current handles and labels from the plot
        handles, labels = plt.gca().get_legend_handles_labels()

        # Create custom handles with thicker lines for the legend
        custom_handles = [Line2D([0], [0], color=handle.get_color(), lw=4, solid_capstyle='round') for handle in handles]

        # Display the legend with custom handles and labels
        plt.legend(
            handles=custom_handles,
            labels=labels,
            fontsize=10,        # Optional: Adjust the font size for the legend labels
            loc='upper center',
            bbox_to_anchor=(0.5, -0.1),
            ncol=6  # You can change this number to adjust the number of columns in the legend
        )
        st.pyplot(fig_4)

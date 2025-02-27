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
import re

import os
import subprocess
import webbrowser
import time

st.set_page_config(
    page_title="Validacija kapacitet CORE - JAO",
    page_icon="☢️",
    layout="wide",  # Options: "centered" (default) or "wide"
    initial_sidebar_state="expanded",  # Options: "auto", "expanded", "collapsed"
)

# Load your data with st.cache_data
@st.cache_data

def load_data():
    # Replace this with your actual dataset
    df_st = pd.DataFrame({
        'date': pd.date_range(start=pd.Timestamp.today().normalize(), periods=1, freq='D'),
    })
    return df_st

# Load data
data = load_data()
st.sidebar.header("Dan validacije") # Sidebar for input
validation_date = st.sidebar.date_input('Izberi dan:',value=data['date'].min())
# Select start and end dates
st.sidebar.header("Obdobje validacije") # Sidebar for input
start_date = st.sidebar.date_input("Začetni dan:", value=data['date'].min())
end_date = st.sidebar.date_input("Končni dan:", value=data['date'].max())

if start_date > end_date:
    st.sidebar.error("Začetni datum mora biti enak ali večji končnemu.")

# Prepare datetime objects
validation_time = datetime.combine(validation_date, datetime.min.time())
start_time = datetime.combine(start_date, datetime.min.time())
end_time = datetime.combine(end_date, datetime.max.time())
tab1, tab2 = st.tabs(["Pretoki in neto pozicije držav", "Informacije o CNE kršitvah"])

def fetch_data_base(url):
    all_data = []
    current_time = validation_time
    from_utc = (current_time - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
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
        st.success("Podatki uspešno pridobljeni!") # Parse the response
    else:
        st.error(f"Napaka pri pridobivanju podatkov: {response.status_code}")
    # Convert to Ljubljana time zone (Europe/Ljubljana)
    df['dateTimeUtc'] = pd.to_datetime(df['dateTimeUtc'], utc=True)
    df['dateTimeUtc'] = df['dateTimeUtc'].dt.tz_convert(pytz.timezone('Europe/Ljubljana'))
    df['dateTimeUtc'] = df['dateTimeUtc'].dt.tz_localize(None)
    df = df.drop(df.columns[0], axis=1) # Nepotreben 'id' stolpec
    return df

def fetch_data_iteration(url,base_param ):
    all_data = []
    # Fetch data from the API
    current_time = validation_time
    while current_time <= validation_time + timedelta(hours=23):
        from_utc = (current_time - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        to_utc = current_time.strftime("%Y-%m-%dT%H:%M:%S.000Z") # + timedelta(hours=1))
        
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
        current_time += timedelta(hours=1) # Kličemo poadtke za vsako uro

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
    # Convert to Ljubljana time zone (Europe/Ljubljana)
    df['dateTimeUtc'] = pd.to_datetime(df['dateTimeUtc'], utc=True)
    df['dateTimeUtc'] = df['dateTimeUtc'].dt.tz_convert(pytz.timezone('Europe/Ljubljana'))
    df['dateTimeUtc'] = df['dateTimeUtc'].dt.tz_localize(None)
    return df
##--------------------------------------------------------------------------- ZAVIHEK 1 ----------------------------------------------------------------------------------------
with tab1:
    st.header("Informacija o pretokih na mejah držav")
    # Display the selected date range
    # st.write(f"#### Izbrano časovno obdobje: {validation_time} - {end_time}")

    # ********************************** PLOT 1 - Prog_ref *************************************
    st.write('### ⚠️ HR-SI = -100 → Hrvaška izvaža v Slovenijo:')
    st.write('### ⚠️ IT-SI = 100 → Italija uvaža iz Slovenije:')
    url = "https://publicationtool.jao.eu/core/api/data/refprog" # Vhodni podatki
    df = fetch_data_base(url) # Klic funkcije za pridobivanje podatkov
    df.columns = df.columns.str.replace('border_', '', regex=False)
    # Seznam vseh držav/povezav
    unique_countries = set()
    for col in df.columns:
        unique_countries.update(col.split('_'))  # Split by '_' and add to the set
    unique_countries = sorted(unique_countries)
    
    #XXXXXXXXXX Drop down seznam XXXXXXXXXXXXXXX 
    default_country = 'SI'
    # selected_country = st.selectbox("Izberi državo", unique_countries)
    selected_country = st.selectbox("Izberi državo:", unique_countries, index=unique_countries.index(default_country))

    # filtered_columns = [col for col in df.columns if selected_country in col]
    pattern = rf'^{selected_country}_[A-Z]+|^[A-Z]+_{selected_country}$'
    filtered_columns = [col for col in df.columns if re.match(pattern, col)]
    df_ref = df[filtered_columns].copy()

    # Dynamically rename columns to remove the selected country
    new_col_names = {col: col.replace('_', '-') for col in filtered_columns}
    df_ref.rename(columns=new_col_names, inplace=True)

    date_str = df['dateTimeUtc'].iloc[0].strftime('%d.%m.%Y')
    color_cycle = cycler('color', plt.cm.tab10.colors)  # # Create a custom color cycle Using the tab20 color map
    plt.rcParams['axes.prop_cycle'] = color_cycle # Apply the color cycle to the current plot

    # Plotting all columns in the df_max DataFrame
    fig_1 = plt.figure(figsize=(12, 6))
    line_width = 2 
    for column in df_ref.columns:
        plt.plot(df['dateTimeUtc'], df_ref[column], label=column, linewidth=line_width, zorder=3)  # Set higher zorder for lines

    # Set the plot title dynamically
    plt.title(f'Izračunane izmenjave pretokov na mejnih povezavah {selected_country} za BD {date_str}')

    # Set the x-axis and y-axis labels
    plt.xlabel('Čas (HH:MM)')
    plt.ylabel('Izmenjane količine [MW]')

    plt.xticks(rotation=0) # Rotate x-axis labels for better readability
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) # Format x-axis to show only hours and minutes (HH:MM)
    plt.grid(True, zorder=0)  # Set lower zorder for grid
    plt.tick_params(axis='x', labelsize=12)  # Adjust labelsize as needed (e.g., 12)

    # Create custom handles with thicker lines for the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    custom_handles = [Line2D([0], [0], color=handle.get_color(), lw=4, solid_capstyle='round') for handle in handles]

    # Display the legend with custom handles and labels
    plt.legend( handles=custom_handles, labels=labels, fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=6)
    st.pyplot(fig_1)
    
    # ********************************** PLOT 2 *************************************
    st.write('### Neto pozicije CORE držav:')
    # ----------------- CORE DRŽAVE ------------------------------
    url = "https://publicationtool.jao.eu/core/api/data/d2CF"
    df = fetch_data_base(url) # Klic funkcije za pridobivanje podatkov
    # Ločimo generacijo/odjem in CORE NET pozicije 
    columns_to_extract_load = [df.columns[0]] + [col for col in df.columns if col.startswith('verticalLoad')]
    columns_to_extract_gen = [df.columns[0]] + [col for col in df.columns if col.startswith('generation')]
    # Določimo df in preimenujemo stolpce
    df_load = df[columns_to_extract_load]; df_gen = df[columns_to_extract_gen]
    df_load.columns = [col.replace('verticalLoad_', '') for col in df_load.columns]; df_gen.columns = [col.replace('generation_', '') for col in df_gen.columns]
    df_net_pos_core = df_gen.iloc[:, 1:] - df_load.iloc[:, 1:]
    df_net_pos_core['dateTimeUtc'] = df_gen.iloc[:, 0] 

    df_net_pos_core['dateTimeUtc'] = pd.to_datetime(df_net_pos_core['dateTimeUtc'])
    df_net_pos_core['Hour_Minute'] = df_net_pos_core['dateTimeUtc'].dt.strftime('%H')
    df_net_pos_core.drop(columns=["dateTimeUtc"], inplace=True)
    df_net_pos_core['Hour_Minute'] = df_net_pos_core['Hour_Minute'].astype(int).astype(str) + ":30"
    df_net_pos_core.set_index('Hour_Minute', inplace=True)
    df_net_pos_core = df_net_pos_core.T
    
    # -----------------NONE CORE DRŽAVE ------------------------------ 
    url = "https://publicationtool.jao.eu/core/api/data/referenceNetPosition"
    df = fetch_data_base(url) # Klic funkcije za pridobivanje podatkov
    columns_to_extract = [df.columns[0]] + [col for col in df.columns if col.startswith('globalNetPosition')]
    df_net_pos_other = df[columns_to_extract]
    df_net_pos_other.columns = [col.replace('globalNetPosition_', '') for col in df_net_pos_other.columns]

    date_str = df['dateTimeUtc'].iloc[0].strftime('%d.%m.%Y')
    df_net_pos_other['dateTimeUtc'] = pd.to_datetime(df_net_pos_other['dateTimeUtc'])
    df_net_pos_other['Hour_Minute'] = df_net_pos_other['dateTimeUtc'].dt.strftime('%H')
    df_net_pos_other.drop(columns=["dateTimeUtc"], inplace=True)
    df_net_pos_other['Hour_Minute'] = df_net_pos_other['Hour_Minute'].astype(int).astype(str) + ":30"
    df_net_pos_other.set_index('Hour_Minute', inplace=True)
    df_net_pos_other = df_net_pos_other.T

    # ----------------- Združi vse države v eno -------------------------
    df_net_pos = pd.concat([df_net_pos_core, df_net_pos_other], axis=0, ignore_index=False)
    df_net_pos = df_net_pos[~df_net_pos.index.isin(['KS'])]
    country_mapping = {
        'AT': 'Austria', 'BE': 'Belgium', 'CZ': 'Czech Republic', 'DE': 'Germany', 'FR': 'France', 
        'HR': 'Croatia', 'HU': 'Hungary', 'NL': 'Netherlands', 'PL': 'Poland', 'RO': 'Romania', 
        'SI': 'Slovenia', 'SK': 'Slovakia', '50Hertz': '50 Hertz', 'Amprion': 'AMPRION', 'Creos': 'Luxembourg',
        'TennetGmbh': 'TennetGmbh', 'Transnet': 'TransnetBW', 'AL': 'Albania', 'BA': 'BiH', 
        'BG': 'Bulgaria', 'CH': 'Switzerland', 'DK1': 'Denmark', 'ES': 'Spain', 
        'GR': 'Greece', 'IT': 'Italy', 'ME': 'Montenegro', 'MK': 'Macedonia', 
        'PT': 'Portugal', 'RS': 'Serbia', 'TR': 'Turkey', 'UA': 'Ukraine', 'KS': 'Kosovo'
    }
    df_net_pos.index = df_net_pos.index.map(country_mapping)

    df_balkan = df_net_pos.loc[['BiH', 'Croatia', 'Serbia', 'Montenegro', 'Macedonia', 'Albania','Turkey', 'Greece', 'Bulgaria'], :] # Izberi države iz Balkana
    df_nib = df_net_pos.loc[['Austria', 'Slovenia', 'Italy', 'Switzerland', 'France'], :] # Izberi države iz NIB
    df_vzhod = df_net_pos.loc[['Hungary', 'Slovakia', 'Czech Republic', 'Poland', 'Ukraine', 'Romania'], :] # Izberi države iz Vzhodne Evrope
    df_sever = df_net_pos.loc[['50 Hertz', 'Netherlands', 'Belgium', 'AMPRION', 'Denmark','TennetGmbh', 'TransnetBW'], :] # Izberi države iz Severne Evrope
    df_iberia = df_net_pos.loc[['Portugal', 'Spain'], :] # Izberi države iz Iberijskega polotoka

    # Neto po regijah
    df_balkan.loc["Neto"] = df_balkan.sum(axis=0).astype(int)
    df_nib.loc["Neto"] = df_nib.sum(axis=0).astype(int)
    df_vzhod.loc["Neto"] = df_vzhod.sum(axis=0).astype(int)
    df_sever.loc["Neto"] = df_sever.sum(axis=0).astype(int)
    df_iberia.loc["Neto"] = df_iberia.sum(axis=0).astype(int)

    # Calculate the target length
    target_length = len("Czech Republic")
    # Rename 'Czech Republic' by padding it with spaces at the beginning
    df_sever = df_sever.rename(index={"TennetGmbh": "  TennetGmbh".rjust(target_length)})
    df_iberia = df_iberia.rename(index={"Portugal": "  Portugal".rjust(target_length+5)})
    df_nib = df_nib.rename(index={"Switzerland": "  Switzerland".rjust(target_length+2)})
    df_balkan = df_balkan.rename(index={"Montenegro": "  Montenegro".rjust(target_length)})
    
    vmin = df_net_pos.min().min()
    vmax = df_net_pos.max().max()

    plt.title(f'Neto pozicije držav za BD {date_str}')
    fig_2, axes = plt.subplots(nrows=5, ncols=1, figsize=(14, 10), gridspec_kw={'height_ratios': [0.6, 0.5, 0.3, 0.5,0.7]})  # Adjust the last value for lower height
    yticksize = 10
    # SEVER heatmap
    sns.heatmap(df_sever, cmap='RdYlGn', annot=True, fmt='.0f', linecolor='black', linewidths=0.1, 
                cbar=False, vmin=vmin, vmax=vmax, ax=axes[0])
    axes[0].set_ylabel('Sever', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('')
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)
    axes[0].xaxis.set_ticks_position('top')
    axes[0].tick_params(axis='y', labelsize=yticksize) 
    y_tick_labels = axes[0].get_yticklabels(); y_tick_labels[-1].set_fontweight('bold'); axes[0].set_yticklabels(y_tick_labels)

    # VZHOD heatmap
    sns.heatmap(df_vzhod, cmap='RdYlGn', annot=True, fmt='.0f', linecolor='black', linewidths=0.1, 
                cbar=False, vmin=vmin, vmax=vmax, ax=axes[1])
    axes[1].set_ylabel('Vzhod', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('')
    axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0)
    axes[1].set_xticklabels([]); axes[1].xaxis.set_ticks([])  # Remove x-axis ticks completely
    axes[1].tick_params(axis='y', labelsize=yticksize) 
    y_tick_labels = axes[1].get_yticklabels(); y_tick_labels[-1].set_fontweight('bold'); axes[1].set_yticklabels(y_tick_labels)

    # IBERIA heatmap
    sns.heatmap(df_iberia, cmap='RdYlGn', annot=True, fmt='.0f', linecolor='black', linewidths=0.1, 
                cbar=False, vmin=vmin, vmax=vmax, annot_kws={"size": 8}, ax=axes[2])
    axes[2].set_ylabel('Iberia', fontsize=12, fontweight='bold')  # Move y-label to the right and change color
    axes[2].set_xlabel('')
    axes[2].set_yticklabels(axes[2].get_yticklabels(), rotation=0)
    axes[2].set_xticklabels([]); axes[2].xaxis.set_ticks([])  # Remove x-axis ticks completely
    axes[2].tick_params(axis='y', labelsize=yticksize) 
    y_tick_labels = axes[2].get_yticklabels(); y_tick_labels[-1].set_fontweight('bold'); axes[2].set_yticklabels(y_tick_labels)

    # NIB heatmap
    sns.heatmap(df_nib, cmap='RdYlGn', annot=True, fmt='.0f', linecolor='black', linewidths=0.1, 
                cbar=False, vmin=vmin, vmax=vmax, annot_kws={"size": 8}, ax=axes[3])
    axes[3].set_ylabel('NIB', fontsize=12, fontweight='bold')
    axes[3].set_xlabel('')
    axes[3].set_yticklabels(axes[3].get_yticklabels(), rotation=0)
    axes[3].set_xticklabels([]); axes[3].xaxis.set_ticks([])  # Remove x-axis ticks completely
    axes[3].tick_params(axis='y', labelsize=yticksize) 
    y_tick_labels = axes[3].get_yticklabels(); y_tick_labels[-1].set_fontweight('bold'); axes[3].set_yticklabels(y_tick_labels)

    # BALKAN heatmap
    sns.heatmap(df_balkan, cmap='RdYlGn', annot=True, fmt='.0f', linecolor='black', linewidths=0.1, 
                cbar=False, vmin=vmin, vmax=vmax, ax=axes[4])
    axes[4].set_ylabel('Balkan', fontsize=12, fontweight='bold')
    axes[4].set_xlabel('')
    axes[4].set_yticklabels(axes[4].get_yticklabels(), rotation=0)
    axes[4].set_xticklabels([]); axes[4].xaxis.set_ticks([])  # Remove x-axis ticks completely
    axes[4].tick_params(axis='y', labelsize=yticksize) 
    y_tick_labels = axes[4].get_yticklabels(); y_tick_labels[-1].set_fontweight('bold'); axes[4].set_yticklabels(y_tick_labels)

    # Create a single color bar for all heatmaps (after the subplots are done)
    cbar_ax = fig_2.add_axes([0.92, 0.1, 0.02, 0.8])  # [left, bottom, width, height] (position the color bar)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)  # Normalize the color range for all subplots
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=norm)
    sm.set_array([])  # Set empty array for color bar display

    # Add the color bar to the figure
    fig_2.colorbar(sm, cax=cbar_ax, label='Neto pozicija [MW]')
    plt.subplots_adjust(hspace=0.05, right=0.9)  # Adjust spacing and allow space for the color bar
    st.pyplot(fig_2)

## --------------------------------------------------------------------------- ZAVIHEK 2 ----------------------------------------------------------------------------------------
with tab2:
    
    st.header("Informacije o CNE kršitvah")
    # Vhodni podatki
    url = "https://publicationtool.jao.eu/core/api/data/finalComputation"
    
    #XXXXXXXXXX Drop down seznam XXXXXXXXXXXXXXX 
    default_TSO = "10XSI-ELES-----1"
    # selected_country = st.selectbox("Izberi državo", unique_countries)
    unique_TSO=["10XAT-APG------Z","10XDE-VE-TRANSMK","10XDE-RWENET---W","10XCZ-CEPS-GRIDE","10XSI-ELES-----1",
            "10XHR-HEP-OPS--A", "10X1001A1001A094","10X1001A1001A329","10XPL-TSO------P","10XFR-RTE------Q",
            "10XSK-SEPS-GRIDB","10X1001A1001A361", "10XDE-ENBW--TNGX","10XDE-EON-NETZ-C","10XRO-TEL------2"]
    selected_TSO = st.selectbox("Izberi TSO:", unique_TSO, index=unique_TSO.index(default_TSO))
    
    base_params = {"Filter": f'{{"Tso": ["{selected_TSO}"]}}', "Skip": 0, "Take": 100}

    df = fetch_data_iteration(url,base_params)
    df = df.drop(['id','tso', 'cneEic','hubFrom','hubTo','substationFrom','substationTo', 'contingencies','elementType','fmaxType','cneStatus', 'presolved','imax','u','justification','contTso','ltaMargin','minRamFactor'], axis=1)
    df = df.iloc[:, :-14]

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

    # Filter options
    filter_iva = st.checkbox("IVA > 0 MW", value=False)
    filter_amr = st.checkbox("AMR > 0 MW", value=False)
    filter_maczt = st.checkbox("MACZT < 70 %", value=True)

    # Apply filters based on user input
    if filter_iva:
        df = df[df['iva'] > 0]
    if filter_amr:
        df = df[df['amr'] > 0]
    if filter_maczt:
        df = df[df['MACZT'] < 70]
   
    # Define font sizes in one place
    font_sizes = {"title": 18,"labels": 16,"ticks": 11,"annot": 14}

    df['Hour_Minute'] = df['dateTimeUtc'].dt.strftime('%H')
    date_str = df['dateTimeUtc'].iloc[0].strftime('%d.%m.%Y')
    # ********************************** PLOT 1 *************************************
    st.write("### MACZT po urah za CNE s kršitvijo")
    heatmap_data = df.pivot_table(columns='info', index='Hour_Minute', values='MACZT', aggfunc='max')

    fig_21 = plt.figure(figsize=(20, 8))
    sns.heatmap(heatmap_data, cmap="YlOrRd_r", annot=True,fmt=".1f", # samo RDYl!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        linewidths=2.5,linecolor='black',cbar_kws={'label': 'MACZT [%]'},
        annot_kws={'size': font_sizes["annot"]})

    plt.title(f"MACZT po urah za CNE s kršitvijo za BD {date_str}", fontsize=font_sizes["title"])
    plt.xlabel("CNE in kršitev", fontsize=font_sizes["labels"])
    plt.ylabel("Ura", fontsize=font_sizes["labels"])
    plt.xticks(rotation=0, fontsize=font_sizes["ticks"])
    plt.yticks(rotation=0, fontsize=font_sizes["ticks"])
    st.pyplot(fig_21)

    # ********************************** PLOT 2 *************************************
    heatmap_data = df.pivot_table(columns='info',index='Hour_Minute', values='iva', aggfunc='max')

    st.write("### IVA po urah za CNE s kršitvijo")
    fig_22 = plt.figure(figsize=(20, 8))
    sns.heatmap(heatmap_data, cmap="YlOrRd", annot=True, fmt=".1f", linewidths=2.5,
        linecolor='black', cbar_kws={'label': 'IVA [MW]'}, annot_kws={'size': font_sizes["annot"]})
    
    plt.title(f"IVA po urah za CNE s kršitvijo za BD {date_str}", fontsize=font_sizes["title"])
    plt.xlabel("CNE in kršitev", fontsize=font_sizes["labels"])
    plt.ylabel("Ura", fontsize=font_sizes["labels"])
    plt.xticks(rotation=0, fontsize=font_sizes["ticks"])
    plt.yticks(rotation=0, fontsize=font_sizes["ticks"])

    st.pyplot(fig_22)

    # Main app layout
    st.write("### Tabela elementov")
    st.dataframe(df, width=1500, height=400)

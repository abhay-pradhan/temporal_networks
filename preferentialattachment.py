import pandas as pd
import igraph
import configparser
import pycountry
from progress.bar import Bar
from progress.spinner import MoonSpinner


G7 = ['United States', 'United Kingdom', 'Germany', 'France', 'Japan', 'Italy','Canada']
BRICS =  ['China', 'Brazil', 'India', 'South Africa', 'Russia']

def get_color(country):
    if country in G7:
        return 'blue'
    elif country in BRICS:
        return 'red'
    else:
        return 'yellow'


def load_dataframe(data_file, df_col, end_timestep):
    df = pd.read_csv(data_file)
    df['timestep'] = df.index
    ret_df = df[df[df_col] < end_timestep]
    original_graph = igraph.Graph.DataFrame(ret_df, directed=False)
    tt = max(ret_df[df_col])
    return ret_df, original_graph, tt


def delete_edges(g, t):
    g_copy = g.copy()
    g_copy.delete_edges(timestep_gt=t)
    return g_copy


def delete_vertices(g, t):
    g_copy = g.copy()
    to_delete_ids = [v.index for v in g_copy.vs if v.degree() == 0]
    g_copy.delete_vertices(to_delete_ids)
    return g_copy


def load_config(config_file):
    cfg = configparser.ConfigParser()
    cfg.read(config_file)
    return cfg


def get_config(cfg, section, key):
    try:
        return cfg[section][key]
    except KeyError:
        print(f'section {section} or key {key} not found')


def get_country(name):
    print(f'Getting country code for {name}')
    try:
        code = pycountry.countries.get(name=name).alpha_2
        return code
    except:
        return name


def format_graph(graph, country_label: False):
    g_copy = graph.copy()
    if country_label:
        g_copy.vs['label'] = [get_country(name) for name in g_copy.vs['name']]

    g_copy.vs['color'] = [get_color(x) for x in g_copy.vs['name']]
    g_copy.vs["size"]  = igraph.rescale(g_copy.betweenness(), (10, 50))
    return g_copy


def bucket_df(df, df_col, bucket_sz):
    interval = pd.interval_range(0, df[df_col].iat[-1], bucket_sz)
    bucketed = df.groupby(pd.cut(df[df_col], bins=interval, right=True)).agg(lambda x: list(x))
    return bucketed


def construct_graph(original_graph, bucketed_df, df_col, country_label):
    old_layout = original_graph.layout_fruchterman_reingold(niter=10, start_temp=0.05, grid='nogrid')
    n = 1
    with MoonSpinner('Generating graph...') as bar:
        for idx, row in bucketed_df.iterrows():
            last_ts_per_bucket = row[df_col][-1]
            g = delete_edges(original_graph, last_ts_per_bucket)
            new_layout = g.layout_fruchterman_reingold(niter=10, start_temp=0.05, grid='nogrid', seed=old_layout)
            gg = format_graph(delete_vertices(g, last_ts_per_bucket), country_label)
            tgt = f'{output_folder}/example{n}.png'
            igraph.plot(gg, layout=new_layout, target=tgt, bbox=(1600,900))
            old_layout = new_layout.copy()
            n += 1
            bar.next()


def generate_betweenness_timeseries(original_graph, df, df_col):
    ret_df = pd.DataFrame()
    with Bar('Generating betweenness...', max=max(df[df_col])) as bar:
        for ts in df[df_col]:
            g = delete_vertices(delete_edges(original_graph, ts), ts)
            df = pd.DataFrame([igraph.rescale(g.betweenness(), (0.1, 1))], columns=[v['name'] for v in g.vs], index=[ts])
            ret_df = ret_df.append(df)
            bar.next()
        return ret_df


def plot_and_save(df, filename):
    df_fig = df.plot().get_figure()
    df_fig.savefig(filename)


cfg = load_config("config.ini")

data_file = get_config(cfg, 'Data', 'data_file')
output_folder = get_config(cfg, 'Data', 'output_folder')

df_col = get_config(cfg, 'Simulation', 'timestep_col_name')
start_ts = int(get_config(cfg, 'Simulation', 'start_timestep'))
end_ts = int(get_config(cfg, 'Simulation', 'end_timestep'))
bucket_sz = int(get_config(cfg, 'Simulation', 'bucket_size'))

country_label = True if cfg['Format']['country_label'] == 'True' else False


print(f'Reading from {data_file} with timestep column name {df_col} start_timestep {start_ts}, end_timestep {end_ts} and bucket_size {bucket_sz} and writing to output_folder {output_folder} with country_labels: {country_label}')
df, original_graph, total_time = load_dataframe(data_file, df_col, end_ts)
print(f'total simulation time is {total_time}')

run_animation = True if cfg['Workflow']['run_animation'] == 'True' else False
run_betweenness = True if cfg['Workflow']['run_betweenness'] == 'True' else False

if run_animation:
    bucketed_df= bucket_df(df, df_col, bucket_sz)
    construct_graph(original_graph, bucketed_df, df_col, country_label)

if run_betweenness:
    between_df = generate_betweenness_timeseries(original_graph, df, df_col)
    between_df.to_csv('betweenness.csv')

    g7_betweenness_df = between_df[G7]
    g7_betweenness_df.to_csv('g7_betweenness.csv')
    plot_and_save(g7_betweenness_df, f'{output_folder}/g7.png'}

    brics_betweenness_df = between_df[BRICS]
    brics_betweenness_df.to_csv('brics_betweenness.csv')
    plot_and_save(g7_betweenness_df, f'{output_folder}/brics.png'}

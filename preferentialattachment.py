import pandas as pd
import igraph
import configparser


def load_dataframe_and_graph(data_file):
    df = pd.read_csv(data_file)
    tt = max(df['time'])
    print(f'total simulation time is {tt}')

    # load it into the graph
    # undirected and use columns as vertex ids
    org_graph = igraph.Graph.DataFrame(df, directed=False, use_vids=True)
    return df, org_graph, tt


# makes a copy of the graph, deletes edges greater than time t and returns the graph
def delete_edges(g, t):
    g_copy = g.copy()
    g_copy.delete_edges(time_gt=t)
    return g_copy


# makes a copy of the graph, deletes vertices with no edges and returns the graph
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


# load configuration from a config.ini file present in current directory
cfg = load_config("config.ini")
data_file= get_config(cfg, 'Simulation', 'data_file')
start_ts = int(get_config(cfg, 'Simulation', 'start_timestep'))
end_ts = int(get_config(cfg, 'Simulation', 'end_timestep'))
bucket_ts = int(get_config(cfg, 'Simulation', 'bucket_timestep'))

print(f'Reading from {data_file} with start_timestep {start_ts}, end_timestep {end_ts} and bucket_timestep {bucket_ts}')
df, original_graph, total_time = load_dataframe_and_graph(data_file)

# make a copy and remove edges that aren't present at simulation start.
start_graph = delete_edges(original_graph, start_ts)

# this is used for the number of iterations while plotting
xs = (x / 10 for x in range(0, total_time*10))

# start with an initial layout
layout_old = start_graph.layout_fruchterman_reingold(niter=10, start_temp=0.05, grid='nogrid')

# start the simulation, keeps a copy of the layout uses it for the next iteration
for i in xs:
    g = delete_edges(original_graph, i)
    layout_new = g.layout_fruchterman_reingold(niter=10, start_temp=0.05, grid='nogrid', seed=layout_old)
    gg = delete_vertices(g, i)
    n = int(i*10)
    tgt = f'animation/example{n}.png'
    print(tgt)
    igraph.plot(gg, layout=layout_new, target=tgt, bbox=(1600,900))
    layout_old = layout_new.copy()



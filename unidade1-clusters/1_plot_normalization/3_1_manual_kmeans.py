from src.utils import load_pokemon_dataset


df = load_pokemon_dataset()


def set_random_cluster_coordinate(num_of_cluster):
    coord_list = []
    label_list = range(0,num_of_cluster)
    #preecher a lista com quatro coordenadas aleatótias.
    return coord_list , label_list

def create_points(df):
    coords = [] # [ [x1,y1], [x2,y2] , [x3,y3]....]
    return coords

def dist_euclidian(p1,p2):
    dist = 0
    return dist

def kmeans(df,num_of_cluster):

    centroids ,  centroids_labels = set_random_cluster_coordinate(num_of_cluster)
    increase_cluster = True
    coords  = create_points(df)
    coord_label = []  #
    while increase_cluster == True:
        for coord in coords:
           coord_label.append(None)
            ##algoritmo

    return  coord_label




num_of_clusters = 2
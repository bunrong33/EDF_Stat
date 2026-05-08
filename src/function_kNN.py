import torch
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import numpy as np
import random

from scipy.stats import boxcox
from statsmodels.tsa.stattools import adfuller, kpss
from pymannkendall import original_test as mk_test
from scipy.spatial.distance import euclidean
from IPython.display import Markdown, display

#------------------------------------------
# 0. Importation et Visualisation
def load_data(base_path: str):
    """
    Charge les données (train, missing, ground truth).

    Paramètres
    ----------
    base_path : dossier contenant les fichiers .pt (ex: "../data/eol_hdf_2021")

    Retourne
    --------
    var_train, covar_train     : données d'entraînement
    var_miss_1, covar_miss_1   : données manquantes scénario 1
    var_miss_2, covar_miss_2   : données manquantes scénario 2
    var_truth, covar_truth     : vérité terrain
    """
    def load(filename):
        data = torch.load(f"{base_path}/{filename}")
        return data[:, 0, :], data[:, 1, :]

    var_train,  covar_train  = load("train_data.pt")
    var_miss_1, covar_miss_1 = load("blocks_missing_1.pt")
    var_miss_2, covar_miss_2 = load("blocks_missing_2.pt")
    var_truth,  covar_truth  = load("ground_truth.pt")

    return var_train, var_miss_1, var_miss_2, var_truth,  covar_train, covar_miss_1, covar_miss_2, covar_truth

def plot_serie (serie_1, serie_2, n1=0, n2=2, titre_1 = "",  ylab_1 = "Consommation d'énergie", label_1 = "",titre_2 = "", ylab_2 = "Température", label_2 =""):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i in range(n1,n2):
        axes[0].plot(serie_1[i,:], label = f"Série {i}" + label_1 )
        axes[1].plot(serie_2[i,:], label = f"Série {i}" + label_2)
    
    axes[0].set_title(titre_1)
    axes[0].set_xlabel("Temps")
    axes[0].set_ylabel(ylab_1)
    axes[0].legend()
    
    axes[1].set_title(titre_2)
    axes[1].set_xlabel("Temps")
    axes[1].set_ylabel(ylab_2)
    axes[1].legend()
    
    plt.show()

# I. Pré-proccessing

# I.0 : Normalisation
def normalize_instance(x, obs_mask):
    """
    x: 1D tensor of shape (T,)
    obs_mask: bool tensor of shape (T,)
    """

    observed = x[obs_mask]
    mu = observed.mean()
    sigma = observed.std()
    x_norm = (x-mu)/(sigma+1e-6)

    return x_norm, mu, sigma

def normalize_all(serie):
    n = serie.shape[0]                   
    mus    = np.zeros(n)
    sigmas = np.zeros(n)
    normed = serie.clone()
    
    for i in range(n):
        normed[i,:], mus[i], sigmas[i] = normalize_instance(
            serie[i,:], 
            ~torch.isnan(serie[i,:])
        )
    
    serie_norm = normed.clone()
    return serie_norm, mus, sigmas

# I.1 : Outliers

def outlier(serie, affiche ="non"):
    serie.clone()
    n, m = serie.shape
    for i in range(n):
        for j in range(3, m - 3):
            median_before = torch.median(torch.stack([serie[i, j-1], serie[i, j-2], serie[i, j-3]]))
            median_after = torch.median(torch.stack([serie[i, j+1], serie[i, j+2], serie[i, j+3]]))
            if torch.abs(serie[i, j]) >= 4 * torch.max(torch.abs(median_before), torch.abs(median_after)):
                serie[i, j] = (serie[i, j-1] + serie[i, j+1]) / 2
                if affiche == "oui":
                    print(f"Suppr. Outlier, serie {i}: t={j}")
    return serie

# I.2 : BOX-COX
def boxcox_transfo(serie):
    n= serie.shape[0]
    serie_boxcox = serie.clone()
    lambda_opt = torch.zeros(n)
    shifts = torch.zeros(n)
    for i in range(n):
        shift = 0
        min_val = serie[i, :][~torch.isnan(serie[i, :])].min()
        if min_val <= 0:
            shift = - min_val + 1e-6 
            # On ajoute à a la série son minimum + epsilon pour n'avoir que des valeurs >0
        shifts[i] = shift
        serie_i = serie[i, :]
        mask_i  = ~torch.isnan(serie_i)
        transformed, lam = boxcox((serie_i[mask_i] + shift).numpy())
        serie_boxcox[i, mask_i] = torch.tensor(transformed, dtype=serie.dtype)
        serie_boxcox[i, ~mask_i] = float('nan')
        lambda_opt[i] = lam
        
    return serie_boxcox, lambda_opt, shifts

def inv_boxcox_transfo(serie_boxcox, lambda_opt, shifts):
    n, m = serie_boxcox.shape
    serie_inv = serie_boxcox.clone()
    for i in range(n):
        lam   = lambda_opt[i]
        shift = shifts[i]
        if lam == 0:
            inv = torch.exp(serie_boxcox[i])
        else:
            base = lam * serie_boxcox[i] + 1
            base = torch.clamp(base, min=1e-10)   # Evite les valeurs négatives sinon on a des "nan" dans l'algo de prédiction
            inv  = base ** (1 / lam)
        serie_inv[i] = inv - shift
    return serie_inv

# I.3 : Gestion de la tendance

def trend_suppr_indiv(serie_i):
    """
    Teste et corrige la tendance d'une série temporelle.
    Retourne : (serie_i_corrigee, type_tendance, méthode_correction)
    """
    # 1 TEST DE MANN-KENDALL : Présence ou non d'une tendance monotone
    serie_i = serie_i.numpy()
    mask = ~np.isnan(serie_i)
    p_mk = mk_test(serie_i[mask]).p
    tendance_detectee = p_mk < 0.05  # Indicatrice(p_mk<0.05)

    if not tendance_detectee :
        return serie_i, "aucune", "aucune"

    # 2 TEST ADF + KPSS: Identification du type de tendance (Stochastique ou déterministe)
    p_adf = adfuller(serie_i[mask], autolag="AIC", regression="ct")[1]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        p_kpss  = kpss(serie_i[mask], nlags="auto", regression = "ct")[1]

    # Racine Unitaire (Stochastique): ADF ne rejette pas H0 et KPSS rejette H0
    stochastique = (p_adf > 0.05) and (p_kpss < 0.05)

    # 3  CORRECTION DE LA TENDANCE
    if stochastique:
        # Tendance stochastique --> Différentiation à lag 1
        serie_i_corrigee = np.concatenate([[0], np.diff(serie_i, n=1)]) # On ajoute un 0 car on perd une dimension sinon
        return serie_i_corrigee, "stochastique", "différenciation"
    else:
        # Tendance déterministe --> Détrending
        t = np.arange(len(serie_i))
        coefs = np.polyfit(t[~np.isnan(serie_i)], serie_i[mask], deg=1)
        tendance = np.polyval(coefs, t)
        serie_i_corrigee = serie_i - tendance
        return serie_i_corrigee, "déterministe", "détrending"


def trend_suppr(serie, affiche="non"):
    n, m = serie.shape
    serie_detrend = serie.clone()
    for i in range(n):
        serie_corr, type_tendance, methode = trend_suppr_indiv(serie[i, :])
        serie_detrend[i, :] = torch.tensor(serie_corr, dtype=serie.dtype)
        if affiche == "oui" and type_tendance != "aucune":
            print(f"serie {i} | Tendance: {type_tendance}, Méthode: {methode}")
    return serie_detrend


def trend_inv(b, type_tendance, methode, gi, T, n, t_start=None):
    """
    Inverse la suppression de tendance sur les prédictions b.
    
    Paramètres:
    -----------
    b             : np.array  - Prédictions dans l'espace détrendé
    type_tendance : str       - "aucune", "stochastique", "déterministe"
    methode       : str       - "aucune", "différenciation", "détrending"
    gi            : np.array  - Série originale (espace Box-Cox, non détrendée)
    T             : int       - Longueur totale de la série
    n             : int       - Nombre de pas prédits
    """
    if t_start is None:
        t_start = T - n
    
    if methode == "aucune":
        return b
    
    elif methode == "différenciation":
        # Reconstruction pas à pas : b contient des incréments
        Y = gi[t_start - 1]
        b_inv = np.zeros(len(b))
        for c in range(len(b)):
            Y = b[c] + Y
            b_inv[c] = Y
        return b_inv
    
    elif methode == "détrending":
        t_all   = np.arange(T)
        valid   = ~np.isnan(gi)
        coefs   = np.polyfit(t_all[valid], gi[valid], deg=1)
        t_test  = np.arange(t_start, t_start + n)
        tendance_test = np.polyval(coefs, t_test)
        return b + tendance_test
    
    else:
        raise ValueError(f"Méthode inconnue : {methode}")

# I.4 : Saisonnalité

def desaisonnaliser(serie_i, methode ="moyenne"):
    """
    Méthode moyenne:
    Soustrait la moyenne sur la saison à toutes les valeurs sur cette saison

    Méthode Diff:
    Différenciation saisonnière lag 336 puis lag 48.
    Conserve la longueur originale avec NaN en tête.
    """
    if methode == "moyenne":
        s1 = torch.full_like(serie_i, float('nan'))
        s2 = torch.full_like(serie_i, float('nan'))
        
        n_weeks = len(serie_i) // 336 
        
        for week in range(n_weeks):
            debut_week = 336 * week
            fin_week   = 336 * (week + 1)
            
            mean_week = serie_i[debut_week:fin_week].mean() 
            s1[debut_week:fin_week] = serie_i[debut_week:fin_week] - mean_week

            for day in range(7):
                debut_day = debut_week + 48 * day
                fin_day   = debut_week + 48 * (day + 1)  

                mean_day = mean_day = torch.nanmean(s1[debut_day:fin_day])  
                s2[debut_day:fin_day] = s1[debut_day:fin_day] - mean_day

        #------------------------------
        reste_debut = 336 * n_weeks

        if reste_debut < len(serie_i):

            # enlever moyenne du reste
            mean_week = torch.nanmean(serie_i[reste_debut:])
            s1[reste_debut:] = serie_i[reste_debut:] - mean_week

            # jours complets dans le reste
            n_days_reste = (len(serie_i) - reste_debut) // 48

            for day in range(n_days_reste):
                debut_day = reste_debut + 48 * day
                fin_day   = reste_debut + 48 * (day + 1)

                mean_day = torch.nanmean(s1[debut_day:fin_day])
                s2[debut_day:fin_day] = s1[debut_day:fin_day] - mean_day

            # reste final (<48 points)
            last = reste_debut + 48 * n_days_reste
            if last < len(serie_i):
                mean_last = torch.nanmean(s1[last:])
                s2[last:] = s1[last:] - mean_last
        #------------------------------

    elif methode == "diff":  
        # Lag 336 (hebdomadaire)
        s1 = torch.full_like(serie_i, float('nan'))
        s1[336:] = serie_i[336:] - serie_i[:-336]

        # Lag 48 (journalier)
        s2 = torch.full_like(s1, float('nan'))
        s2[384:] = s1[384:] - s1[384-48:-48] 

    return s2

def desaisonnaliser_all(serie, methode):
    n, m = serie.shape
    serie_desaison = serie.clone()
    for i in range(n):
        s = desaisonnaliser(serie[i, :],methode)
        serie_desaison[i, :] = s.detach().clone()
    return serie_desaison

def inv_desaisonnaliser(b, serie_i, T, n, methode="moyenne"):
    """
    Inverse la désaisonnalisation sur les prédictions b.
    
    Paramètres:
    -----------
    b        : np.array      - Prédictions désaisonnalisées (longueur n)
    serie_i  : torch.tensor  - Série originale complète (longueur T)
    T        : int           - Longueur totale
    n        : int           - Nombre de pas prédits
    methode  : str           - "moyenne" ou "diff"
    """
    b_inv = b.copy()
    
    if methode == "moyenne":
        # Ré-ajouter la moyenne de la semaine et du jour correspondants
        for c in range(n):
            t = T - n + c  # indice dans la série originale
            
            # Semaine correspondante
            week = t // 336
            debut_week = 336 * week
            fin_week   = 336 * (week + 1)
            mean_week  = serie_i[debut_week:fin_week].mean().item()
            
            # Jour correspondant dans la semaine
            day = (t % 336) // 48
            debut_day = debut_week + 48 * day
            fin_day   = debut_week + 48 * (day + 1)
            s1_day    = serie_i[debut_day:fin_day] - mean_week
            mean_day  = s1_day.mean().item()
            
            b_inv[c] = b[c] + mean_week + mean_day

    elif methode == "diff":
        # Inverser lag 48 puis lag 336
        for c in range(n):
            t = T - n + c
            # Ré-ajouter lag 48
            if t - 48 >= 0:
                b_inv[c] = b[c] + serie_i[t - 48].item()
        for c in range(n):
            t = T - n + c
            # Ré-ajouter lag 336
            if t - 336 >= 0:
                b_inv[c] = b_inv[c] + serie_i[t - 336].item()

    return b_inv

# I.--
def plot_preprocessing(var, var_norm, var_outliers, var_bc, var_detrend, var_deseason,
                       n1=0, n2=1,
                       titre_var="", var_ylab="Consommation d'énergie", var_lab=""):
    
    titres = ["Original", "Normalisé", "Sans outliers", "Box-Cox", "Détrendé", "Désaisonnalisé"]
    var_types = [var, var_norm, var_outliers, var_bc, var_detrend, var_deseason]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 6))
    fig.suptitle(titre_var, fontsize=14)
    
    for i in range(n1, n2):
        for idx, (ax, data, titre) in enumerate(zip(axes.flatten(), var_types, titres)):
            ax.plot(data[i, :], label=f"Série {i}" + var_lab)
            ax.set_xlabel("Temps")
            ax.set_ylabel(var_ylab)
            ax.set_title(titre)
            ax.legend()
    
    plt.tight_layout()
    plt.show()




# II Modélisation k-NN et choix des paramètres (k,p)

def find_blocks(mask_i):
    """
    Trouve les blocks manquants

    Paramètres:
    -----------
    mask_i   : array  :  booléen indiquant les NaN d'une série

    Retourne:
    ---------
    retourne : array :  Début et fin des blocks manquants [(start1, end1), (start2, end2), ...]
    """
    blocks = []
    in_block = False
    for idx, val in enumerate(mask_i):
        if val and not in_block:
            start = idx
            in_block = True
        elif not val and in_block:
            end = idx - 1
            blocks.append((start, end))
            in_block = False
    if in_block:
        blocks.append((start, len(mask_i) - 1))
    return blocks

def predi_knn_general(x_start, k, p, x_truth):
    """
    Prévision par k-NN sur séries temporelles.
    
    Paramètres:
    -----------
    x_start  : torch.Tensor  - Jeu de données avec valeurs manquantes (N x T)
    k        : int           - Nombre de voisins
    p        : int           - Longueur de la fenêtre de recherche
    x_truth  : torch.Tensor  - Jeu de données complet (vérité terrain) (N x T)
     
    Retourne:
    ---------
    mape_tab      : np.array         - MAPE pour chaque série (N,)
    f_predi_torch : torch.Tensor     - Prédictions sur toute la série (N x T), NaN hors zones manquantes
    mae_tab       : np.array         - MAE pour chaque série (N,)
    mae_norm_tab  : np.array         - MAE normalisée pour chaque série (N,)
    mask          : torch.BoolTensor - Masque des valeurs manquantes (N x T)
    x_test_norm   : np.array         - Valeurs réelles normalisées sur les zones manquantes (N x n)
    """
    N, T = x_start.shape
    mask = x_start.isnan()
    n_per_serie = mask.sum(dim=1)
    n = n_per_serie[0].item()
    
    x_norm, mus, sigmas = normalize_all(x_start)
    x_norm_out = outlier(x_norm);
    
    x_test = x_truth[mask].reshape(N, -1).numpy()
    x_truth_norm = (x_truth - torch.tensor(mus).unsqueeze(1)) / (torch.tensor(sigmas).unsqueeze(1) + 1e-6)
    x_test_norm = x_truth_norm[mask].reshape(N, -1).numpy()
    #------
    
    x = x_norm_out.clone().detach()
    x_train = x.clone()                     
    
    g, lam, shi = boxcox_transfo(x_train)
    g ,lam, shi  = g.numpy(), lam.numpy(), shi.numpy()
    
    mape_tab = np.zeros(N)
    mae_tab = np.zeros(N)
    mae_norm_tab = np.zeros(N)
    nb_blocs    = np.zeros(N)
    f_all = [np.full(T, np.nan) for _ in range(N)]


    
def predi_knn_general(x_start, k, p, x_truth):
    """
    Prévision par k-NN sur séries temporelles.
    
    Paramètres:
    -----------
    x_start  : torch.Tensor  - Jeu de données avec valeurs manquantes (N x T)
    k        : int           - Nombre de voisins
    p        : int           - Longueur de la fenêtre de recherche
    x_truth  : torch.Tensor  - Jeu de données complet (vérité terrain) (N x T)
     
    Retourne:
    ---------
    mape_tab      : np.array         - MAPE pour chaque série (N,)
    f_predi_torch : torch.Tensor     - Prédictions sur toute la série (N x T), NaN hors zones manquantes
    mae_tab       : np.array         - MAE pour chaque série (N,)
    mae_norm_tab  : np.array         - MAE normalisée pour chaque série (N,)
    mask          : torch.BoolTensor - Masque des valeurs manquantes (N x T)
    x_test_norm   : np.array         - Valeurs réelles normalisées sur les zones manquantes (N x n)
    """
    N, T = x_start.shape
    mask = x_start.isnan()
    n_per_serie = mask.sum(dim=1)
    n = n_per_serie[0].item()
    
    x_norm, mus, sigmas = normalize_all(x_start)
    x_norm_out = outlier(x_norm);
    
    x_test = x_truth[mask].reshape(N, -1).numpy()
    x_truth_norm = (x_truth - torch.tensor(mus).unsqueeze(1)) / (torch.tensor(sigmas).unsqueeze(1) + 1e-6)
    x_test_norm = x_truth_norm[mask].reshape(N, -1).numpy()
    #------
    
    x = x_norm_out.clone().detach()
    x_train = x.clone()                     
    
    g, lam, shi = boxcox_transfo(x_train)
    g ,lam, shi  = g.numpy(), lam.numpy(), shi.numpy()
    
    mape_tab = np.zeros(N)
    mae_tab = np.zeros(N)
    mae_norm_tab = np.zeros(N)
    nb_blocs    = np.zeros(N)
    f_all = [np.full(T, np.nan) for _ in range(N)]
    
    for i in range(N):
        ni = n_per_serie[i].item()    # nb de valeurs manquantes pour la série i
        nan_mask_i = mask[i].numpy()
        if np.isnan(g[i][~nan_mask_i]).any():
            print(f"Série {i} : nan dans g après Box-Cox (hors valeurs manquantes)")
            continue
        gi = g[i]
        serie_corr, type_tendance, methode = trend_suppr_indiv(torch.tensor(g[i], dtype=torch.float64))
        #serie_deseason = desaisonnaliser(torch.tensor(g[i], dtype=torch.float64), methode ="moyenne")
        hi = serie_corr
        #hi = serie_deseason.numpy()

        # trouver les indices des block manquant
        nan_blocks = find_blocks(mask[i].numpy())
        for block in nan_blocks:
            t_start, t_end = block
            ni = t_end - t_start + 1
            fenetre_requete = hi[t_start - p : t_start]
            if len(fenetre_requete) != p or np.isnan(fenetre_requete).any():
                #si pas assez d'observations avant le premier trou, on prend la fenetre d'apres comme fenetre_requete
                fenetre_requete = hi[t_end + 1  : t_end + 1 + p]
                if len(fenetre_requete) != p or np.isnan(fenetre_requete).any():
                    # Si aussi le cas pour celle la on l'ignore dans la recherche
                    print(f"Série {i}, bloc {t_start}-{t_end} : impossible de construire une fenêtre de taille p={p}")
                    continue
        
            # Candidats avant et après le "trou"
            D = {}
            # Avant le bloc
            for j in range(0, t_start - p + 1):
                fenetre_candidate = hi[j : j + p]
                if np.isnan(fenetre_candidate).any() or len(fenetre_candidate) != p:
                    continue
                # Exclure si un des futurs tombe dans le trou
                if any(t_start <= j + p + c <= t_end for c in range(ni)):
                    continue
                D[j] = euclidean(fenetre_requete, fenetre_candidate) # Calcul des distances (dist euclidienne)
            
            # Après le bloc
            for j in range(t_end + 1, T - p + 1):
                fenetre_candidate = hi[j : j + p]
                if np.isnan(fenetre_candidate).any() or len(fenetre_candidate) != p:
                    continue
                D[j] = euclidean(fenetre_requete, fenetre_candidate)     # Calcul des distances (dist euclidienne)
        
            Dm = sorted(D.items(), key=lambda x: x[1])    # Ordonne les fenetres 
            voisins = []     # Ensemble des k plus proches voisins = Ensembles des fenêtres candidates retenues
            voisins_distances = {}
        
            for rank in range(k):
                j, dist = Dm[rank]           # On prend le rank-ième plus proche voisin
                voisins.append(j)            # On stocke l'indice de la fenêtre
                voisins_distances[j] = dist  # On stocke la distance associée
        
            Y = gi[t_start - 1]      # Dernier point avant le "trou"
        
            # Prévisions sur les n derniers instants
            b = np.zeros(ni)   # init des valeurs à prédire sur la zone de test
        
            for c in range(ni): 
                weights = {}
                for j in voisins:
                    d = voisins_distances[j]
                    weights[j] = 1.0 / d if d > 0 else 1e10   # Poids = 1/distance
                total_weight = sum(weights.values())
        
                sc = 0.0
                valid = 0
                for j in voisins:
                    future_idx = j + p + c               # Indice de la différence future du voisin j
                    if future_idx < len(hi) and not np.isnan(hi[future_idx]):              # Vérification des bornes
                        sc += (weights[j] / total_weight) * hi[future_idx]   # Moyenne Pondérée
                        valid += 1
    
                if valid == 0:
                    bc = Y
                        
                elif methode == "différenciation":
                    bc = sc + Y    # Ajoute la moyenne pondérée de (1/distance) à la dernière valeur connue ou prédite
                    Y  = bc        
                else:
                    bc = sc        
                b[c] = bc   
                
            #b = inv_desaisonnaliser(b, torch.tensor(g[i], dtype=torch.float64), T-n, n, methode="moyenne")
    
            if methode != "différenciation":
                b = trend_inv(b, type_tendance, methode, gi, T, ni, t_start=t_start)
                
            if np.isnan(b).any() or np.isinf(b).any():
                print(f"Série {i}, bloc {t_start}-{t_end} : nan/inf dans b après trend_inv")
                continue
            
            if lam[i] != 0 and np.any(b * lam[i] + 1 <= 0):
                print(f"Série {i} : valeurs négatives dans b avant inv_boxcox (lam={lam[i]:.3f}, min={np.min(b * lam[i] + 1):.4f})")
                # Clamp pour éviter le nan
                b = np.clip(b, -1/lam[i] + 1e-6 if lam[i] > 0 else None, None)
            
            f = inv_boxcox_transfo(
                torch.tensor(b,        dtype=x.dtype).unsqueeze(0),  # (n,) → (1, n)
                torch.tensor([lam[i]], dtype=x.dtype),               # (1,)
                torch.tensor([shi[i]], dtype=x.dtype)                # (1,)
            ).numpy().flatten()                                      # (1, n) → (n,)
            
            if np.isnan(f).any():
                print(f"Série {i} : nan dans f après inv_boxcox")
                continue
            
    
            f_denorm = f * (sigmas[i] + 1e-6) + mus[i] # Dénormalisation
            
            #Erreur sur chacun des block
            x_test_block      = x_truth[i, t_start:t_end+1].numpy()
            x_test_norm_block = x_truth_norm[i, t_start:t_end+1].numpy()

            f_all[i][t_start:t_end+1] = f_denorm  

            denom = np.where(x_test_block == 0, np.nan, x_test_block)
            mape_tab[i]     += np.nanmean(np.abs((f_denorm - x_test_block) / denom)) * 100
            mae_tab[i]      += np.mean(np.abs(f_denorm - x_test_block))
            mae_norm_tab[i] += np.mean(np.abs(f - x_test_norm_block))
            nb_blocs[i]     += 1

        # Moyenne sur tous les blocs de la série
        if nb_blocs[i] > 0:
            mape_tab[i]     /= nb_blocs[i]
            mae_tab[i]      /= nb_blocs[i]
            mae_norm_tab[i] /= nb_blocs[i]

    f_predi_torch = torch.tensor(np.stack(f_all), dtype=torch.float64)

    return mape_tab, f_predi_torch, mae_tab, mae_norm_tab , mask, x_test_norm   #, b, f_denorm, mus, sigmas, g, x*


def opti_kp(x_train,k_min= 3 , k_max = 7, I = 1, n = 48, nb_day_missing=1):
    """
    Optimisation du choix de la taille de la fenêtre (p) et du nombre de voisin (k)
    
    Paramètres:
    -----------
    x_train    : torch.tensor   - Jeu de données d'entrainement complet
           #  p_limit    : int           - Taille maximale de la fenêtre
    k_min      : int           - Nombre min de voisins
    k_max      : int           - Nombre max de voisins

    I          : int           - Nombre de jeu d'entrainement
    n          : int           - Nombre de données manquantes à simuler (1 jour = 1 bloc de 48)
     
    Retourne:
    ---------
    k_opti   : int/array    - k qui minimise la MAE (ou MAPE (à voir))
    p_opti   : int/array    - p qui minimise la MAE (ou MAPE (à voir))
    mae_opti : int/array    - MAE obtenu avec ces paramètre optimisés
    """       
    T, N = x_train.shape
    obs_per_day = N//28
    mae_norm_opti = 1000   # joue le rôle de la MAE optimale au départ
    k_opti, p_opti = None, None
    
    p_values = [4, 8, 12, 24, 48]  # Fenetre de  2,4,6,12,24 hrs
    k_values = list(range(k_min, k_max + 1))

    for k in k_values:
        for p in p_values:
            m = np.zeros(I)
            for i in range(1, I + 1):
                x_train_miss = x_train.clone()

                jours_possibles = list(range(1,27))  #Evite de supprimer le 1er ou le dernier jour
                jours_choisis = random.sample(jours_possibles, nb_day_missing)

                for jour_manquant in jours_choisis:
                    debut = jour_manquant * obs_per_day
                    x_train_miss[:, debut : debut + obs_per_day] = float('nan')
                
                mape_tab, f_predi_torch, mae_tab, mae_norm_tab , mask, x_test_norm = predi_knn_general(x_train_miss, k, p, x_train)
                m[i-1] = np.mean(mae_norm_tab)
            
            mae_norm_moy = np.mean(m)
            if mae_norm_moy < mae_norm_opti:
                mae_norm_opti = mae_norm_moy
                k_opti = k
                p_opti = p
                print(f"p = {p}, k = {k}, mae = {mae_norm_opti:.3f}\n")
        print(f"----- Fin test {k} voisins -----")
    return k_opti, p_opti, mae_norm_opti



def plot_predictions(k1, k2, mask, f, x_miss, x_truth):
    """
    Affiche les prédictions complètes et par bloc de NaN pour chaque série.

    Paramètres
    ----------
    k1, k2       : indices de début et fin des séries à afficher
    mask         : masque (tensor) des NaN, shape (N, T)
    f            : prédictions, shape (N, T)
    x_miss       : séries avec valeurs manquantes, shape (N, T)
    x_truth      : séries réelles, shape (N, T)
    """
    nan_blocks_all = [find_blocks(mask[i].numpy()) for i in range(k1, k2)]
    nb_blocs_max = max(len(b) for b in nan_blocks_all)

    fig, axes = plt.subplots(
        k2 - k1, 1 + nb_blocs_max,
        figsize=(8 * (1 + nb_blocs_max), 6 * (k2 - k1)),
        gridspec_kw={'width_ratios': [2] + [1] * nb_blocs_max}
    )

    # Garantit que axes est toujours 2D
    if (k2 - k1) == 1:
        axes = axes[np.newaxis, :]

    for i in range(k1, k2):
        row = i - k1
        pred_full = f[i]

        # Graphe complet
        axes[row, 0].plot(x_miss[i, :], label="Manquante", color='black')
        axes[row, 0].plot(pred_full, label="Prédiction", linestyle="--")
        axes[row, 0].plot(x_truth[i, :], label="Réelle", color='green', linewidth=0.5)
        axes[row, 0].legend()
        axes[row, 0].set_title(f"Série {i} complète")

        # Un graphe par bloc
        nan_blocks = nan_blocks_all[row]
        for b_idx, (t_start, t_end) in enumerate(nan_blocks):
            idx = np.arange(t_start, t_end + 1)
            ax = axes[row, 1 + b_idx]
            ax.plot(idx, x_truth[i, idx], label="Réelle", color='green')
            ax.plot(idx, pred_full[idx], label="Prédiction", linestyle="--", color='orange')
            ax.legend()
            ax.set_title(f"Série {i} — Bloc {b_idx+1} (t={t_start}:{t_end})")

        # Masquer les axes vides
        for b_idx in range(len(nan_blocks), nb_blocs_max):
            axes[row, 1 + b_idx].set_visible(False)

    plt.tight_layout()
    plt.show()





def to_float(x):
    if isinstance(x, np.ndarray):
        return float(np.nanmean(x))
    return float(x)

def affiche_tableau(rows):
    lines = []
    lines.append("| Dataset | Setting | TabPFN | Ridge on Covar | k-NN |")
    lines.append("|:--------|:--------|-------:|---------------:|------:|")
    for dataset, block, tabpfn, ridge, k, p, mae, bold in rows:
        tabpfn_str = f"{tabpfn:.3f}" if tabpfn is not None else "-"
        ridge_str  = f"{ridge:.3f}"  if ridge  is not None else "-"
        knn_str    = f"{to_float(mae):.3f} (k={int(to_float(k))},p={int(to_float(p))})"
        if bold:
            knn_str = f"**{knn_str}**"
        lines.append(f"| {dataset} | *{block}* | {tabpfn_str} | {ridge_str} | {knn_str} |")
    display(Markdown("\n".join(lines)))

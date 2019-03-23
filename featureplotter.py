#Function that takes an h5 file containing raw data as an input. For each feature of the input data, will plot a histogram 
#of the feature in the ttbar vs. tW data sets and save them as pdfs. Returns a list of features that have statistically significant 
#(in this case <.05) p-value outputs of the Kolmogorov-Smirnov statistic to be used when creating x_train and x_test sets from raw data
#to be fed into a neural network.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import twaml
from scipy import stats
import operator

def featurePlotter(tWfilepath, ttbarfilepath):
    tW1j1b = twaml.dataset.from_pytables(tWfilepath, 'tW_DR_1j1b')
    ttbar1j1b = twaml.dataset.from_pytables(ttbarfilepath, 'ttbar_1j1b') #imports ttbar and tW datasets
    tWdf = tW1j1b.df 
    ttdf = ttbar1j1b.df #turns the ttbar and tW data into respective Pandas dataframes
    
    remove = ['sumet', 'mass_jetL1', 'E_jetL1', 'mT_jet1met', 'mass_jet1met', 'deltaR_jet1_met',
          'H_jet1met', 'mass_jet1', 'mT_lep1met', 'mass_jetF', 'eom_lep1met', 'mass_lep2met',
          'pTsys_lep2met', 'pT_jetF', 'pT_jet1', 'cent_lep1met', 'eom_jet1met', 'pT_lep1', 
          'pT_lep2', 'phi_jetL1', 'pTsys_lep1met', 'met', 'HT_jet1met', 'eta_lep1', 'E_jetF',
          'E_jet1', 'H_lep2met', 'HT_lep2met', 'pTsys_jet1met', 'eta_jet1', 'eta_jetF',
          'eta_jetL1', 'eta_jet1met', 'cent_jet1met', 'reg2j1b', 'reg2j2b', 'reg2j0b', 'reg2j2bLmm',
          'reg2j2bHmm', 'runNumber', 'mv2c10_jet1', 'mv2c10_jetF', 'elmu', 'mumu', 'elel',
          'pdgId_lep1', 'pdgId_lep2', 'charge_lep1', 'charge_lep2', 'mass_lep1', 'mass_lep2',
          'psuedoContTagBin_jet1', 'psuedoContTagBin_jetF', 'reg1j1b', 'reg1j0b', 'OS', 'SS',
          'randomRunNumber', 'eventNumber', 'njets', 'nbjets', 'nloosebjets', 'minimaxmbl',
          'eta_lep2met', 'eta_jetAvg', 'E_lep2', 'mT_lep2met', 'HT_lep1met', 'H_lep1met', 'mass_lep1met',
         'eom_lep2met', 'E_lep1', 'eta_lep1met', 'DL1_jetF', 'sigpTsys_jet1met', 'cent_lep2met',
          'eta_lep2', 'sigpTsys_lep1met', 'sigpTsys_lep2met', 'deltaphi_lep1_met', 'deltaR_lep1_met',
          'deltaphi_lep2_met', 'deltapT_lep1_met'] #creating a list of variables not of interest. Includes systematic prone variables (single object variables, variables containing non-kinematic information etc.)
    for name in ttdf.columns:
    if 'jet2' in name:
        remove.append(name) #adds kinematic variables pertaining to jet2 to this list of variables of no interest
    ttdf = ttdf.drop(columns=remove) #removes columns containing variables not of interest from the dataframe
    
    list = []
    for name in ttdf.columns: #iterates through each feature in the data. The ttbar and tW data sets should have the same features in each so choosing which datatframe's columns to iterate over is not an issue
        tt = ttdf[name].to_numpy() 
        tW = tWdf[name].to_numpy() #transforms each column into numpy arrays
        if (name,(stats.ks_2samp(tW, tt)[1] < .05:
                  list.append((name,(stats.ks_2samp(tW, tt)[1])) #appends a tuple to the list of the form (feature name, p-value output of the Kolmogorov-Smirnov statistic) if the p-value is statistically significant (in this case p <.05)
        f = plt.figure()
        plt.hist([tt, tW], bins=50, label = ["tt", "tW"], histtype="step", density=True) #plots histograms of the feature between the ttbar and tW data sets
        plt.legend()
        plt.title(name)
        plt.show()
        save = name + ".pdf"
        f.savefig(save, bbox_inches='tight') #saves each histogram as a pdf titled with the name of the feature
        
    list.sort(key=operator.itemgetter(1)) #sorts the list by the p-values of each feature
    return list #returns the features that are statistically significant. This list will be used to create the x_train and x_test sets from raw data.

# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Scenario evolving impulse response functions with fair
#
# with different shapes over the historical
#
# the total emissions size will be 1 MtCH4, and this will be emitted over different time frames over 30 years on top of ssp119, ssp245 and ssp585

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import xarray as xr

import fair
from fair import FAIR
from fair.interface import fill, initialise
from fair.io import read_properties

# %%
fair.__version__

# %% [markdown]
# ## First, a historical all-forcings run

# %%
f = FAIR(ch4_method='Thornhill2021')
f.define_time(1750, 2500, 1)
scenarios = ['ssp119', 'ssp245', 'ssp585']
f.define_scenarios(scenarios)

# %%
fair_params_1_4_0 = '../data/fair-calibration/calibrated_constrained_parameters_1.4.0.csv'
df_configs = pd.read_csv(fair_params_1_4_0, index_col=0)
configs = df_configs.index  # label for the "config" axis
f.define_configs(configs)

# %%
species, properties = read_properties(filename='../data/fair-calibration/species_configs_properties_1.4.0.csv')
f.define_species(species, properties)

# %%
f.allocate()

# %%
f.fill_from_csv(
    forcing_file='../data/forcing/volcanic_solar.csv',
)

# %%
# I was lazy and didn't convert emissions to CSV, so use the old clunky method of importing from netCDF
# this is from calibration-1.4.0
da_emissions = xr.load_dataarray("../data/emissions/ssp_emissions_1750-2500.nc")
output_ensemble_size = 841
da = da_emissions.loc[dict(config="unspecified", scenario=scenarios)]
fe = da.expand_dims(dim=["config"], axis=(2))
f.emissions = fe.drop_vars(("config")) * np.ones((1, 1, output_ensemble_size, 1))

# %%
fill(
    f.forcing,
    f.forcing.sel(specie="Volcanic") * df_configs["forcing_scale[Volcanic]"].values.squeeze(),
    specie="Volcanic",
)
fill(
    f.forcing,
    f.forcing.sel(specie="Solar") * df_configs["forcing_scale[Solar]"].values.squeeze(),
    specie="Solar",
)

f.fill_species_configs("../data/fair-calibration/species_configs_properties_1.4.0.csv")
f.override_defaults("../data/fair-calibration/calibrated_constrained_parameters_1.4.0.csv")

# initial conditions
initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

# %%
f.run()

# %%
output = f.temperature.sel(layer=0).data.transpose((1,0,2))
output.shape

# %%
ds = xr.Dataset(
    data_vars = dict(
        temperature = (['scenario', 'timebounds', 'config'], output),
    ),
    coords = dict(
        scenario = ['ssp119', 'ssp245', 'ssp585'],
        timebounds = np.arange(1750, 2501),
        config = df_configs.index
    ),
    attrs = dict(units = 'K')
)

# %%
ds

# %%
ds.to_netcdf('../output/base_scenarios.nc')

# %% [markdown]
# ## Create some emissions profiles

# %%
perturbations = {}
profiles = ['frontload', 'backload', 'bell', 'constant', 'rampup', 'rampdown', 'constant10']
norm_shape = scipy.stats.norm.pdf(scipy.stats.norm.ppf(np.arange(0.5, 30)/30))
int_norm_shape = np.sum(norm_shape)
normalisation = 1 / int_norm_shape

for scenario in scenarios:
    perturbations[scenario] = {}
    for profile in profiles:
        perturbations[scenario][profile] = np.zeros(475)
    perturbations[scenario]['frontload'][0] = 1
    perturbations[scenario]['backload'][29] = 1
    perturbations[scenario]['constant'][0:30] = 1/30
    perturbations[scenario]['constant10'][0:30] = 10/30
    perturbations[scenario]['rampup'][0:30] = np.arange(0.5/30, 30/30, 1/30) * 2 / 30
    perturbations[scenario]['rampdown'][0:30] = np.arange(29.5/30, 0/30, -1/30) * 2 / 30
    perturbations[scenario]['bell'][0:30] = norm_shape * normalisation

# %%
for profile in profiles:
    plt.plot(np.arange(0.5, 30), perturbations["ssp245"][profile][:30])

# %% [markdown]
# ## Perturbation runs

# %%
f_irf = {}
for profile in profiles:
    new_emissions = f.emissions.copy()
    new_emissions[275:, 0, :, 3] = new_emissions[275:, 0, :, 3] + perturbations["ssp119"][profile][:, None]
    new_emissions[275:, 1, :, 3] = new_emissions[275:, 1, :, 3] + perturbations["ssp245"][profile][:, None]
    new_emissions[275:, 2, :, 3] = new_emissions[275:, 2, :, 3] + perturbations["ssp585"][profile][:, None]
    
    f_irf[profile] = FAIR(ch4_method='Thornhill2021')
    f_irf[profile].define_time(1750, 2500, 1)
    f_irf[profile].define_scenarios(scenarios)
    f_irf[profile].define_configs(configs)
    f_irf[profile].define_species(species, properties)
    f_irf[profile].allocate()
    f_irf[profile].fill_from_csv(
        forcing_file='../data/forcing/volcanic_solar.csv',
    )
    f_irf[profile].emissions = new_emissions
    fill(
        f_irf[profile].forcing,
        f_irf[profile].forcing.sel(specie="Volcanic") * df_configs["forcing_scale[Volcanic]"].values.squeeze(),
        specie="Volcanic",
    )
    fill(
        f_irf[profile].forcing,
        f_irf[profile].forcing.sel(specie="Solar") * df_configs["forcing_scale[Solar]"].values.squeeze(),
        specie="Solar",
    )
    
    f_irf[profile].fill_species_configs("../data/fair-calibration/species_configs_properties_1.4.0.csv")
    f_irf[profile].override_defaults("../data/fair-calibration/calibrated_constrained_parameters_1.4.0.csv")
    
    # initial conditions
    initialise(f_irf[profile].concentration, f_irf[profile].species_configs["baseline_concentration"])
    initialise(f_irf[profile].forcing, 0)
    initialise(f_irf[profile].temperature, 0)
    initialise(f_irf[profile].cumulative_emissions, 0)
    initialise(f_irf[profile].airborne_emissions, 0)
    
    f_irf[profile].run()

# %% [markdown]
# ### IRF is the difference of the run with and without the extra 1Mt (or 10Mt)CH4
#
# Sense check: IRFs on page 17 of https://www.ipcc.ch/site/assets/uploads/2018/07/WGI_AR5.Chap_.8_SM.pdf
#
# note this is one model with a higher ECS than the AR6 assessment, so really this is bang in line

# %%
# the IRFs are the differences between the runs with an additional 1 tCO2 and the base scenarios.
#'frontload', 'backload', 'bell', 'constant', 'rampup', 'rampdown'
irf = {}
for scenario in scenarios:
    irf[scenario] = {}
    for profile in profiles:
        irf[scenario][profile] = (f_irf[profile].temperature-f.temperature).sel(scenario=scenario, layer=0, timebounds=np.arange(2024, 2501))

# %%
#irf

# %%
os.makedirs('../plots', exist_ok=True)

# %%
fig, ax = plt.subplots(1, 3, figsize=(16, 6))
for iscen, scenario in enumerate(scenarios):
    for profile in profiles:
        if profile == "constant10":
            scaling = 1/10
            label = "constant10 / 10"
        else:
            scaling = 1
            label = profile
        ax[iscen].plot(np.arange(-1, 476), irf[scenario][profile].median(dim='config') * scaling, label=label);
    ax[iscen].set_title(f'Impulse response to 1 MtCH4 upon {scenario}')
    ax[iscen].set_xlim(0, 100)
    ax[iscen].set_ylim(0.0, 7e-5)
#plt.axhline(0, ls=":", color="k")
ax[0].set_ylabel('Temperature increase, K')
ax[0].legend()

plt.savefig('../plots/shapes_ch4.png')

# %%
fig, ax = plt.subplots(1, 3, figsize=(16, 6))
for iscen, scenario in enumerate(scenarios):
    ax[iscen].plot(np.arange(-1, 476), (100*(irf[scenario]["constant10"]/10-irf[scenario]["constant"])/irf[scenario]["constant"]).median(dim='config'));
    ax[iscen].set_title(f'% difference 10MtCH4/10 v. 1MtCH4 for {scenario}')
    ax[iscen].set_xlim(0, 100)
#    ax[iscen].set_ylim(0.0, 7e-5)
#plt.axhline(0, ls=":", color="k")
ax[0].set_ylabel('% relative to 1MtCH4')

plt.savefig('../plots/difference_10Mt_1Mt_ch4.png')

# %%
#xr.DataArray.from_dict(irf)

# %%
output = np.stack(
    (
        np.stack(
            (
                irf['ssp119']['frontload'].data, 
                irf['ssp119']['backload'].data, 
                irf['ssp119']['bell'].data,
                irf['ssp119']['constant'].data, 
                irf['ssp119']['rampup'].data, 
                irf['ssp119']['rampdown'].data,
                irf['ssp119']['constant10'].data,
            ), axis=0
        ), 
        np.stack(
            (
                irf['ssp245']['frontload'].data, 
                irf['ssp245']['backload'].data, 
                irf['ssp245']['bell'].data,
                irf['ssp245']['constant'].data, 
                irf['ssp245']['rampup'].data, 
                irf['ssp245']['rampdown'].data,
                irf['ssp245']['constant10'].data,
            ), axis=0
        ),
        np.stack(
            (
                irf['ssp585']['frontload'].data, 
                irf['ssp585']['backload'].data, 
                irf['ssp585']['bell'].data,
                irf['ssp585']['constant'].data, 
                irf['ssp585']['rampup'].data, 
                irf['ssp585']['rampdown'].data,
                irf['ssp585']['constant10'].data,
            ), axis=0
        ),
    ), axis=0
)

# %%
#output

# %%
ds = xr.Dataset(
    data_vars = dict(
        temperature = (['scenario', 'profile', 'timebounds', 'config'], output),
    ),
    coords = dict(
        scenario = ['ssp119', 'ssp245', 'ssp585'],
        profile = ['frontload', 'backload', 'bell', 'constant', 'rampup', 'rampdown', 'constant10'],
        timebounds = np.arange(-1, 476),
        config = df_configs.index
    ),
    attrs = dict(units = 'K')
)

# %%
ds

# %%
os.makedirs('../output/', exist_ok=True)

# %%
ds.to_netcdf('../output/shapes_irf_1MtCH4.nc')

# %%
f_irf['frontload']

# %%
(f_irf[profile].temperature).sel(layer=0).shape

# %%
# reuse output array 
output = np.stack(
    (
        (f_irf['frontload'].temperature).sel(layer=0).data, 
        (f_irf['backload'].temperature).sel(layer=0).data, 
        (f_irf['bell'].temperature).sel(layer=0).data,
        (f_irf['constant'].temperature).sel(layer=0).data, 
        (f_irf['rampup'].temperature).sel(layer=0).data, 
        (f_irf['rampdown'].temperature).sel(layer=0).data,
        (f_irf['constant10'].temperature).sel(layer=0).data,
    ), axis=2
).transpose(1,2,0,3)

# %%
ds = xr.Dataset(
    data_vars = dict(
        temperature = (['scenario', 'profile', 'timebounds', 'config'], output),
    ),
    coords = dict(
        scenario = ['ssp119', 'ssp245', 'ssp585'],
        profile = ['frontload', 'backload', 'bell', 'constant', 'rampup', 'rampdown', 'constant10'],
        timebounds = np.arange(1750, 2501),
        config = df_configs.index
    ),
    attrs = dict(units = 'K')
)

# %%
ds.to_netcdf('../output/shapes_raw_1MtCH4.nc')

# %%

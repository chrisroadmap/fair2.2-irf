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
# the total emissions size will be 30 Mt, and this will be emitted over different time frames over 30 years on top of ssp245

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
scenarios = ['ssp245']
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
da = da_emissions.loc[dict(config="unspecified", scenario=["ssp245"])]
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

# fill(f_stoch.climate_configs['stochastic_run'], True)
# fill(f_stoch.climate_configs['use_seed'], True)

# initial conditions
initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

# %%
f.run()

# %% [markdown]
# ## Create some emissions profiles

# %%
perturbations = {}
scenarios = ['frontload', 'backload', 'bell', 'constant', 'rampup', 'rampdown']
for scenario in scenarios:
    perturbations[scenario] = np.zeros(475)
perturbations['frontload'][0] = 30
perturbations['backload'][29] = 30
perturbations['constant'][0:30] = 1
perturbations['rampup'][0:30] = np.arange(0.5, 30) * 2 / 30
perturbations['rampdown'][0:30] = np.arange(29.5, 0, -1) * 2 / 30

# %%
norm_shape = scipy.stats.norm.pdf(scipy.stats.norm.ppf(np.arange(0.5, 30)/30))
int_norm_shape = np.sum(norm_shape)
normalisation = 30 / int_norm_shape
perturbations['bell'][0:30] = norm_shape * normalisation

# %%
plt.plot(np.arange(0.5, 30),perturbations['bell'][0:30])

# %%
for scenario in scenarios:
    plt.plot(np.arange(0.5, 30), perturbations[scenario][:30])

# %% [markdown]
# ## Perturbation runs

# %%
f_irf = {}
for scenario in scenarios:
    new_emissions = f.emissions.copy()
    new_emissions[275:, :, :, 3] = new_emissions[275:, :, :, 3] + perturbations[scenario][:, None, None]
    
    f_irf[scenario] = FAIR(ch4_method='Thornhill2021')
    f_irf[scenario].define_time(1750, 2500, 1)
    f_irf[scenario].define_scenarios(['ssp245'])
    f_irf[scenario].define_configs(configs)
    f_irf[scenario].define_species(species, properties)
    f_irf[scenario].allocate()
    f_irf[scenario].fill_from_csv(
        forcing_file='../data/forcing/volcanic_solar.csv',
    )
    f_irf[scenario].emissions = new_emissions
    fill(
        f_irf[scenario].forcing,
        f_irf[scenario].forcing.sel(specie="Volcanic") * df_configs["forcing_scale[Volcanic]"].values.squeeze(),
        specie="Volcanic",
    )
    fill(
        f_irf[scenario].forcing,
        f_irf[scenario].forcing.sel(specie="Solar") * df_configs["forcing_scale[Solar]"].values.squeeze(),
        specie="Solar",
    )
    
    f_irf[scenario].fill_species_configs("../data/fair-calibration/species_configs_properties_1.4.0.csv")
    f_irf[scenario].override_defaults("../data/fair-calibration/calibrated_constrained_parameters_1.4.0.csv")
    
    # initial conditions
    initialise(f_irf[scenario].concentration, f_irf[scenario].species_configs["baseline_concentration"])
    initialise(f_irf[scenario].forcing, 0)
    initialise(f_irf[scenario].temperature, 0)
    initialise(f_irf[scenario].cumulative_emissions, 0)
    initialise(f_irf[scenario].airborne_emissions, 0)
    
    f_irf[scenario].run()

# %% [markdown]
# ### IRF is the difference of the run with an additional 1 MtCH4 pulse in 2024
#
# Sense check: IRFs on page 17 of https://www.ipcc.ch/site/assets/uploads/2018/07/WGI_AR5.Chap_.8_SM.pdf
#
# note this is one model with a higher ECS than the AR6 assessment, so really this is bang in line

# %%
# the IRFs are the differences between the runs with an additional 1 tCO2 and the base scenarios.
#'frontload', 'backload', 'bell', 'constant', 'rampup', 'rampdown'
irf = {}
for scenario in scenarios:
    irf[scenario] = (f_irf[scenario].temperature-f.temperature).sel(scenario='ssp245', layer=0, timebounds=np.arange(2024, 2501))

# %%
irf

# %%
os.makedirs('../plots', exist_ok=True)

# %%
for scenario in scenarios:
# plt.fill_between(
#     np.arange(-1, 475),
#     irf_ssp245.min(dim='config'), 
#     irf_ssp245.max(dim='config'), 
#     color='#f69320', 
#     alpha=0.2
# );
# plt.fill_between(
#     np.arange(-1, 475),
#     irf_ssp245.quantile(.05, dim='config'), 
#     irf_ssp245.quantile(.95, dim='config'), 
#     color='#f69320', 
#     alpha=0.2
# );
# plt.fill_between(
#     np.arange(-1, 475),
#     irf_ssp245.quantile(.16, dim='config'), 
#     irf_ssp245.quantile(.84, dim='config'), 
#     color='#f69320', 
#     alpha=0.2
# );
    plt.plot(np.arange(-1, 476), irf[scenario].median(dim='config'), label=scenario);
plt.xlim(0, 100)
plt.ylim(-0.0e-4, 18e-4)
#plt.axhline(0, ls=":", color="k")
plt.ylabel('Temperature increase, K')
plt.title('Impulse response to 30 MtCH4 upon ssp245')
plt.legend()

plt.savefig('../plots/shapes_ch4.png')

# %%
output = np.stack(
    (
        irf['frontload'].data, 
        irf['backload'].data, 
        irf['bell'].data,
        irf['constant'].data, 
        irf['rampup'].data, 
        irf['rampdown'].data,
    ),
axis=0)

# %%
ds = xr.Dataset(
    data_vars = dict(
        temperature = (['scenario', 'timebound', 'config'], output),
    ),
    coords = dict(
        scenario = ['frontload', 'backload', 'bell', 'constant', 'rampup', 'rampdown'],
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
ds.to_netcdf('../output/shapes_irf_30MtCH4.nc')

# %%

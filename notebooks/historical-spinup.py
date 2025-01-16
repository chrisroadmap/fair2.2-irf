# ---
# jupyter:
#   jupytext:
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

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

da_emissions = xr.load_dataarray("../data/emissions/ssp_emissions_1750-2500.nc")
output_ensemble_size = 841
da = da_emissions.loc[dict(config="unspecified", scenario=["ssp119", "ssp245", "ssp585"])]
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
# ## Perturbation run

# %%
new_emissions = f.emissions.copy()
new_emissions[274, :, :, 0] = new_emissions[274, :, :, 0] + 1e-9

# %%
f_irf = FAIR(ch4_method='Thornhill2021')
f_irf.define_time(1750, 2500, 1)
scenarios = ['ssp119', 'ssp245', 'ssp585']
f_irf.define_scenarios(scenarios)
f_irf.define_configs(configs)
f_irf.define_species(species, properties)
f_irf.allocate()
f_irf.fill_from_csv(
    forcing_file='../data/forcing/volcanic_solar.csv',
)
f_irf.emissions = new_emissions
fill(
    f_irf.forcing,
    f_irf.forcing.sel(specie="Volcanic") * df_configs["forcing_scale[Volcanic]"].values.squeeze(),
    specie="Volcanic",
)
fill(
    f_irf.forcing,
    f_irf.forcing.sel(specie="Solar") * df_configs["forcing_scale[Solar]"].values.squeeze(),
    specie="Solar",
)

f_irf.fill_species_configs("../data/fair-calibration/species_configs_properties_1.4.0.csv")
f_irf.override_defaults("../data/fair-calibration/calibrated_constrained_parameters_1.4.0.csv")

# fill(f_stoch.climate_configs['stochastic_run'], True)
# fill(f_stoch.climate_configs['use_seed'], True)

# initial conditions
initialise(f_irf.concentration, f_irf.species_configs["baseline_concentration"])
initialise(f_irf.forcing, 0)
initialise(f_irf.temperature, 0)
initialise(f_irf.cumulative_emissions, 0)
initialise(f_irf.airborne_emissions, 0)

f_irf.run()

# %% [markdown]
# ### IRF is the difference of the run with an additional 1 tCO2 pulse in 2024
#
# Sense check: IRFs on page 17 of https://www.ipcc.ch/site/assets/uploads/2018/07/WGI_AR5.Chap_.8_SM.pdf
#
# note this is one model with a higher ECS than the AR6 assessment, so really this is bang in line

# %%
plt.plot((f_irf.temperature-f.temperature).sel(scenario='ssp245', layer=0, timebounds=np.arange(2024, 2501)));
plt.xlim(0, 200)
plt.ylim(-0.1e-12, 2e-12)

# %%
plt.plot((f_irf.temperature-f.temperature).sel(scenario='ssp119', layer=0, timebounds=np.arange(2024, 2501)));
plt.xlim(0, 20)
plt.ylim(-0.1e-12, 1e-12)

# %%
plt.plot((f_irf.temperature-f.temperature).sel(scenario='ssp585', layer=0, timebounds=np.arange(2024, 2501)));
plt.xlim(0, 100)
plt.ylim(-0.1e-12, 1e-12)

# %% [markdown]
# ## Second, a future CO2-only run spun up from present day climate

# %%
f = FAIR()
f.define_time(2025, 3026, 1)
scenarios = ['irf-1tonCO2']
f.define_scenarios(scenarios)
f.define_configs(configs)

# %%
species = ['CO2', 'CH4', 'N2O']
properties = {
    "CO2": {
        'type': 'co2',
        'input_mode': 'emissions',
        'greenhouse_gas': True,
        'aerosol_chemistry_from_emissions': False,
        'aerosol_chemistry_from_concentration': False
    },
    "CH4": {
        'type': 'ch4',
        'input_mode': 'emissions',
        'greenhouse_gas': True,
        'aerosol_chemistry_from_emissions': False,
        'aerosol_chemistry_from_concentration': False
    },
    "N2O": {
        'type': 'n2o',
        'input_mode': 'emissions',
        'greenhouse_gas': True,
        'aerosol_chemistry_from_emissions': False,
        'aerosol_chemistry_from_concentration': False
    }
}

# %%
f.define_species(species, properties)
f.allocate()

# %%
f.concentration.loc[dict(specie='CH4')] = 808.2490285
f.concentration.loc[dict(specie='N2O')] = 273.021047

# %%
emission_co2 = np.zeros(1001)
emission_co2[0] = 1e-9 # gigaton CO2 to ton CO2

# %%
f.emissions

# %%
f.emissions.loc[dict(specie='CO2')] = emission_co2[:, None, None]
fill(f.emissions, 0, specie='CH4')
fill(f.emissions, 0, specie='N2O')

# %%
f.fill_species_configs("../data/fair-calibration/species_configs_properties_1.4.0.csv")
f.override_defaults("../data/fair-calibration/calibrated_constrained_parameters_1.4.0.csv")

# %%
conc.sel(specie='CO2')

# %%
gasp

# %%
# initiliase from startdump
initialise(f.concentration, conc[-1, :, :, 2:5].values)
initialise(f.forcing, forc[-1, :, :, 2:5].values)
initialise(f.temperature, temp[-1, ...].values)
initialise(f.airborne_emissions, abem[-1, :, :, 2:5].values)
initialise(f.cumulative_emissions, cuem[-1, :, :, 2:5].values)
initialise(f.ocean_heat_content_change, ohcc[-1, ...].values)
f.gas_partitions=gasp[-1, :, 2:5, :].values

# run
f.run()

# %%
f.concentration.shape

# %%
conc[-1, :, :, 2:5].shape

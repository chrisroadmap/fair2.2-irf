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

# %% [markdown]
# # Scenario evolving impulse response functions with fair
#
# For numerical stability, we want to make the pulse sizes reasonable, so add 1 MtCH4 in 2025 (about 0.25-0.3% of current CH4 emissions).

# %%
import os

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
# this is from calibration-1.4.0
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
new_emissions[274, :, :, 3] = new_emissions[274, :, :, 3] + 1

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
# ### IRF is the difference of the run with an additional 1 MtCH4 pulse in 2024
#
# Sense check: IRFs on page 17 of https://www.ipcc.ch/site/assets/uploads/2018/07/WGI_AR5.Chap_.8_SM.pdf
#
# note this is one model with a higher ECS than the AR6 assessment, so really this is bang in line

# %%
# the IRFs are the differences between the runs with an additional 1 tCO2 and the base scenarios.

irf_ssp119 = (f_irf.temperature-f.temperature).sel(scenario='ssp119', layer=0, timebounds=np.arange(2025, 2501))
irf_ssp245 = (f_irf.temperature-f.temperature).sel(scenario='ssp245', layer=0, timebounds=np.arange(2025, 2501))
irf_ssp585 = (f_irf.temperature-f.temperature).sel(scenario='ssp585', layer=0, timebounds=np.arange(2025, 2501))

# %%
irf_ssp245

# %%
os.makedirs('../plots', exist_ok=True)

# %%
plt.fill_between(
    np.arange(-1, 475),
    irf_ssp245.min(dim='config'), 
    irf_ssp245.max(dim='config'), 
    color='#f69320', 
    alpha=0.2
);
plt.fill_between(
    np.arange(-1, 475),
    irf_ssp245.quantile(.05, dim='config'), 
    irf_ssp245.quantile(.95, dim='config'), 
    color='#f69320', 
    alpha=0.2
);
plt.fill_between(
    np.arange(-1, 475),
    irf_ssp245.quantile(.16, dim='config'), 
    irf_ssp245.quantile(.84, dim='config'), 
    color='#f69320', 
    alpha=0.2
);
plt.plot(np.arange(-1, 475), irf_ssp245.median(dim='config'), color='#f69320');
plt.xlim(0, 100)
plt.ylim(-0.1e-4, 1.2e-4)
plt.axhline(0, ls=":", color="k")
plt.ylabel('Temperature increase, K')
plt.title('Impulse response to 1 MtCH4 upon ssp245')

plt.savefig('../plots/ssp245_ch4.png')

# %%
plt.fill_between(
    np.arange(-1, 475),
    irf_ssp119.min(dim='config'), 
    irf_ssp119.max(dim='config'), 
    color='#00a9cf', 
    alpha=0.2
);
plt.fill_between(
    np.arange(-1, 475),
    irf_ssp119.quantile(.05, dim='config'), 
    irf_ssp119.quantile(.95, dim='config'), 
    color='#00a9cf', 
    alpha=0.2
);
plt.fill_between(
    np.arange(-1, 475),
    irf_ssp119.quantile(.16, dim='config'), 
    irf_ssp119.quantile(.84, dim='config'), 
    color='#00a9cf', 
    alpha=0.2
);
plt.plot(np.arange(-1, 475), irf_ssp119.median(dim='config'), color='#00a9cf');
plt.xlim(0, 100)
plt.ylim(-0.1e-4, 1.2e-4)
plt.axhline(0, ls=":", color="k")
plt.ylabel('Temperature increase, K')
plt.title('Impulse response to 1 MtCH4 upon ssp119')

plt.savefig('../plots/ssp119_ch4.png')

# %%
plt.fill_between(
    np.arange(-1, 475),
    irf_ssp585.min(dim='config'), 
    irf_ssp585.max(dim='config'), 
    color='#980002', 
    alpha=0.2
);
plt.fill_between(
    np.arange(-1, 475),
    irf_ssp585.quantile(.05, dim='config'), 
    irf_ssp585.quantile(.95, dim='config'), 
    color='#980002', 
    alpha=0.2
);
plt.fill_between(
    np.arange(-1, 475),
    irf_ssp585.quantile(.16, dim='config'), 
    irf_ssp585.quantile(.84, dim='config'), 
    color='#980002', 
    alpha=0.2
);
plt.plot(np.arange(-1, 475), irf_ssp585.median(dim='config'), color='#980002');
plt.xlim(0, 100)
plt.ylim(-0.1e-4, 1.2e-4)
plt.axhline(0, ls=":", color="k")
plt.ylabel('Temperature increase, K')
plt.title('Impulse response to 1 MtCH4 upon ssp585')

plt.savefig('../plots/ssp585_ch4.png')

# %%
plt.fill_between(
    np.arange(-1, 475),
    (irf_ssp585-irf_ssp245).min(dim='config'), 
    (irf_ssp585-irf_ssp245).max(dim='config'), 
    color='k', 
    alpha=0.2
);
plt.fill_between(
    np.arange(-1, 475),
    (irf_ssp585-irf_ssp245).quantile(.05, dim='config'), 
    (irf_ssp585-irf_ssp245).quantile(.95, dim='config'), 
    color='k', 
    alpha=0.2
);
plt.fill_between(
    np.arange(-1, 475),
    (irf_ssp585-irf_ssp245).quantile(.16, dim='config'), 
    (irf_ssp585-irf_ssp245).quantile(.84, dim='config'), 
    color='k', 
    alpha=0.2
);
plt.plot(np.arange(-1, 475), (irf_ssp585-irf_ssp245).median(dim='config'), color='k');
plt.xlim(0, 475)
plt.ylim(-1.2e-4, 1.2e-4)
plt.ylabel('Temperature increase, K')
plt.title('Difference ssp585 to ssp245')

plt.savefig('../plots/diff_ssp585_ssp245_ch4.png')

# %%
plt.fill_between(
    np.arange(-1, 475),
    (irf_ssp245-irf_ssp119).min(dim='config'), 
    (irf_ssp245-irf_ssp119).max(dim='config'), 
    color='k', 
    alpha=0.2
);
plt.fill_between(
    np.arange(-1, 475),
    (irf_ssp245-irf_ssp119).quantile(.05, dim='config'), 
    (irf_ssp245-irf_ssp119).quantile(.95, dim='config'), 
    color='k', 
    alpha=0.2
);
plt.fill_between(
    np.arange(-1, 475),
    (irf_ssp245-irf_ssp119).quantile(.16, dim='config'), 
    (irf_ssp245-irf_ssp119).quantile(.84, dim='config'), 
    color='k', 
    alpha=0.2
);
plt.plot(np.arange(-1, 475), (irf_ssp245-irf_ssp119).median(dim='config'), color='k');
plt.xlim(0, 475)
plt.ylim(-1.2e-4, 1.2e-4)
plt.ylabel('Temperature increase, K')
plt.title('Difference ssp245 to ssp119')

plt.savefig('../plots/diff_ssp245_ssp119_ch4.png')

# %%
output = np.stack((irf_ssp119.data, irf_ssp245.data, irf_ssp585.data), axis=0)
output.shape

# %%
ds = xr.Dataset(
    data_vars = dict(
        temperature = (['scenario', 'timebound', 'config'], output),
    ),
    coords = dict(
        scenario = ['irf_ssp119', 'irf_ssp245', 'irf_ssp585'],
        timebounds = np.arange(-1, 475),
        config = df_configs.index
    ),
    attrs = dict(units = 'K/MtCH4')
)

# %%
ds

# %%
os.makedirs('../output/', exist_ok=True)

# %%
ds.to_netcdf('../output/irf_1MtCH4.nc')

# %%

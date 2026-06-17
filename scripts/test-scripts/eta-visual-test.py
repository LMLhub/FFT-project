from fft_project.cue_class import Cue
from fft_project.decision_class import FFT
from fft_project.experiment_class import Experiment
from fft_project.cue_features import avoid_worst_n_ranks, growth_rate, expected_isoelastic_utility
import pandas as pd

from fft_project.simulation_gamble_data import simulate_gamble_data
from fft_project.helper_scripts.create_cues_ffts import create_cues, create_ffts
#Fractal values for the additive dynamic from the experiment.
FRACTAL_VALUES = [-407.0, -305.5, -241.5, -49.0, 50.0, 108.5, 210.5, 309.5, 440.5]
FRACTAL_VALUES_MULTI = [-0.850, -0.5395, -0.433, -0.1735, 0.006, 0.1685, 0.369, 0.5535, 0.772]

def create_ffts():
    fft = FFT(id="fft2",
              name="Avoid the worst or random - additive",
              description="An example FFT avoid the worst.",
              cues=[Cue.cue_registry["c03"]])

    growth_rate_fft = FFT(id="fft3",
              name="Growth rate maximisation",
              description="An example FFT with the growth rate cue.",
              cues=[Cue.cue_registry["c02"]])

    growth_rate_multi_fft = FFT(id="fft4",
              name="Avoid the worst - multiplicative",
              description="An example FFT with avoid the worst cue.",
              cues=[Cue.cue_registry["c04"]])

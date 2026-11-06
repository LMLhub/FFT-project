import sys
import logging
import copy
import numpy as np

from fft_project.config import read_dotenv, parse_args, read_config_file, setup_logging
from fft_project.config import setup_run_id, setup_output_folders, save_config
from fft_project.simulation_gamble_data import simulate_gamble_data
from fft_project.create_cues_ffts import create_cues2
from fft_project.cue_class import Cue
from fft_project.sequential_decision_rules.data import Dataset
from fft_project.sequential_decision_rules.metrics import evaluate_features
from fft_project.sequential_decision_rules.search import SearchState, validate_state, extend_state
from fft_project.sequential_decision_rules.pareto import pareto_frontier


logger = logging.getLogger(__name__)


def initialize_run():
  args = parse_args()
  # Set up logging based on the provided log level
  setup_logging(loglevel=args.loglevel)
  logging.info(f"Using configuration file: {args.config}")

  # Load environment variables from .env file
  remote_drive = read_dotenv()["REMOTE_DRIVE"]
  logging.info(f"Local location of shared remote drive: {remote_drive}")

  # Read config file
  config = read_config_file(args.config)
  logging.info(f"Configuration loaded: {config}")

  # Set up run ID
  config["run_id"] = setup_run_id(args.run_id)

  # Set up output folders
  setup_output_folders(remote_drive, config["run_id"])

  # Save config file to the run's input folder
  save_config(args.config, remote_drive, config["run_id"])

  return args, config, remote_drive


def build_prediction_matrix_and_labels(
  gamble_data,
  all_cues,
  cue_args,
  priority_heuristic_tolerance,
  priority_heuristic_tolerance_no_loss,
):
  # Create prediction matrix and labels
  cues_to_consider = list(all_cues.keys())
  # Remove GR-max from the cues to be considered since this will be used to
  # generate the labels
  cues_to_consider.remove("GR-max")
  cues_to_consider.remove("GR-min-minus")
  n_rows = len(gamble_data)
  n_cues = len(cues_to_consider)
  logger.info(f"Number of training instances = {n_rows}")
  logger.info(f"Number of candidate cues = {n_cues}")
  y = np.zeros(n_rows, dtype=int)
  x = np.zeros((n_rows, n_cues), dtype=int)

  for i, row in gamble_data.iterrows():
    # Get the gamble parameters
    g_l_up = row["gamma_left_up"]
    g_l_down = row["gamma_left_down"]
    g_r_up = row["gamma_right_up"]
    g_r_down = row["gamma_right_down"]
    # Generate the entry i of the labels vector, y
    # Selecting GR-max cue for generating the labels
    cue = all_cues["GR-max"]
    extra_args = {}
    _, side_if_true = cue.evaluate(g_l_up, g_l_down, g_r_up, g_r_down, **extra_args)
    if side_if_true == "left":
      y[i] = 1
    elif side_if_true == "right":
      y[i] = -1
    else:
      raise RuntimeError(f"Reference cue {cue.id} failed to assign a choice for row {i}.")
    cue_identifiers = {}
    # Generate the row i of the prediction matrix, x
    for j, cue_label in enumerate(cues_to_consider):
      cue = all_cues[cue_label]
      cue_identifiers[j] = cue.id
      # Read what additional arguments the current cue requires
      extra_arg_names = cue.required_args[4:] if getattr(cue, "required_args", None) else []
      extra_args = {}
      effective_cue_args = dict(cue_args)
      if cue.id in ["PH-1", "PH-3"]:
        effective_cue_args["tol"] = priority_heuristic_tolerance
      elif cue.id in ["PH-1-NL", "PH-3-NL"]:
        effective_cue_args["tol"] = priority_heuristic_tolerance_no_loss
      for name in extra_arg_names:
        if name in row.index:
          extra_args[name] = row[name]
        elif name in effective_cue_args:
          extra_args[name] = effective_cue_args[name]
        else:
          raise ValueError(f"Missing required argument '{name}' for cue '{cue.id}' in row {i}.")
      _, side_if_true = cue.evaluate(g_l_up, g_l_down, g_r_up, g_r_down, **extra_args)
      if side_if_true == "left":
        x[i, j] = 1
      elif side_if_true == "right":
        x[i, j] = -1
      else:
        x[i, j] = 0

  return x, y, cue_identifiers


def simulate_dataset(config):
  n_samples = config["gamble_simulation"]["n"]
  if config["gamble_simulation"]["dynamic"] == "multiplicative":
    dynamic = "multiplicative"
    fractal_values = config["gamble_simulation"]["fractals_mul"]
    priority_heuristic_tolerance = 0.35
    priority_heuristic_tolerance_no_loss = 0.2
    gamble_data = simulate_gamble_data(n_samples, config["gamble_simulation"]["fractals_mul"], random_seed = 42)
  elif config["gamble_simulation"]["dynamic"] == "additive":
    dynamic = "additive"
    fractal_values = config["gamble_simulation"]["fractals_add"]
    gamble_data = simulate_gamble_data(n_samples, config["gamble_simulation"]["fractals_add"], random_seed = 42)
    priority_heuristic_tolerance = 0.5
    priority_heuristic_tolerance_no_loss = 0.4
  else:
    raise ValueError(f"Invalid dynamic type: {config['gamble_simulation']['dynamic']}. Must be 'multiplicative' or 'additive'.")
  # add a wealth column to gamble_data for testing purposes
  gamble_data["wealth"] = np.random.randint(410, 810, size=len(gamble_data))
  return (
    gamble_data,
    dynamic,
    fractal_values,
    priority_heuristic_tolerance,
    priority_heuristic_tolerance_no_loss,
  )


def set_up_cues(config):
  # Set up cues and set up the cue registry
  create_cues2(config)

  # Copy the cue registry to a dict
  all_cues = Cue.cue_registry

  # Create the complement of each cue and add it to the list for
  # subsequent evaluation
  complement_cues = {}
  for (id, cue) in all_cues.items():
    new_id = id + "-minus"
    complement_cues[new_id] = copy.deepcopy(cue)
    complement_cues[new_id].id = new_id
    complement_cues[new_id].direction = - cue.direction
  for (id, cue) in complement_cues.items():
    all_cues[id] = cue

  return all_cues


def main():
  print("Running sequential decision rules test script")
  _, config, _ = initialize_run()

  (
    gamble_data,
    dynamic,
    fractal_values,
    priority_heuristic_tolerance,
    priority_heuristic_tolerance_no_loss,
  ) = simulate_dataset(config)

  # This is where all the additional arguments required by some Cue objects
  # are set
  cue_args = {"dynamic": dynamic,
              "eta": 0.5,
              "fractal_values": fractal_values,
              "n": 2,
              "tol" : priority_heuristic_tolerance
              }


  all_cues = set_up_cues(config)

  X, Y, cue_identifiers = build_prediction_matrix_and_labels(
    gamble_data,
    all_cues,
    cue_args,
    priority_heuristic_tolerance,
    priority_heuristic_tolerance_no_loss,
  )

  # Start sequential decision rule evaluation from here
  dataset = Dataset(X, Y)
  print(f"{dataset.N} instances, {dataset.M} features")
  initial_state = SearchState(
    sequence=(),
    remaining=np.ones(dataset.N, dtype=bool)
  )
  validate_state(initial_state, dataset)
  metrics = evaluate_features(
    dataset,
    initial_state.remaining
  )

  print(metrics)

  frontier = pareto_frontier(metrics)

  print(f"Total features: {len(metrics)}")
  print(f"Pareto-optimal features: {len(frontier)}")

  print(frontier)

  import matplotlib.pyplot as plt

  fig, ax = plt.subplots(figsize=(7, 5))

  ax.scatter(
      metrics["coverage"],
      metrics["accuracy"],
      alpha=0.4,
      label="All features"
  )
  for idx, row in metrics.iterrows():
    ax.annotate(
      cue_identifiers[idx],
      (row["coverage"], row["accuracy"]),
      xytext=(4, 4),
      textcoords="offset points",
      fontsize=8,
      alpha=0.8,
    )

  ax.plot(
      frontier["coverage"],
      frontier["accuracy"],
      "o-",
      label="Pareto frontier"
  )

  ax.set_xlabel("Coverage")
  ax.set_ylabel("Accuracy")
  ax.set_title("Feature performance and Pareto frontier")
  ax.set_xlim(0, 1.05)
  ax.set_ylim(0, 1.05)
  ax.grid(alpha=0.3)
  ax.legend()

  plt.show()

  feature = int(frontier.iloc[0]["feature"])

  child = extend_state(
    initial_state,
    feature,
    dataset
  )

  print("Selected feature:", feature)
  print("Sequence:", child.sequence)
  print("Correct:", child.correct)
  print("Incorrect:", child.incorrect)
  print("Remaining:", child.remaining.sum())

  row = metrics.loc[metrics["feature"] == feature].iloc[0]

  assert child.correct == row["correct"]
  assert child.incorrect == row["incorrect"]
  assert child.remaining.sum() == dataset.N - row["classified"]

  assert initial_state.sequence == ()
  assert initial_state.remaining.all()
  assert initial_state.correct == 0
  assert initial_state.incorrect == 0

  validate_state(child, dataset)

  residual_metrics = evaluate_features(
    dataset,
    child.remaining
  )

  residual_frontier = pareto_frontier(
    residual_metrics[
        ~residual_metrics["feature"].isin(child.sequence)
    ]
  )
  feature2 = int(residual_frontier.iloc[0]["feature"])

  grandchild = extend_state(
    child,
    feature2,
    dataset
  )

  print("Selected feature:", feature2)
  print("Sequence:", grandchild.sequence)
  print("Correct:", grandchild.correct)
  print("Incorrect:", grandchild.incorrect)
  print("Remaining:", grandchild.remaining.sum())
  validate_state(grandchild, dataset)

  assert grandchild.remaining.sum() <= child.remaining.sum()
  assert grandchild.correct >= child.correct
  assert grandchild.incorrect >= child.incorrect
  return 0


if __name__ == "__main__":
  returncode = main()
  sys.exit(returncode)

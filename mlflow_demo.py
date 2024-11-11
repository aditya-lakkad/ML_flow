"""
Authore = aditya lakkad
Date = 08th Nov,2024
Description = This python file create dummy workflow to refer entire
work flow of MLflow
Usage = python mlflow_demo.py --param1 <input1> --param2 <input2>
"""

import mlflow
import os
import argparse  # to get user input
import time


def mlflow_tracking(p1: int, p2: int) -> None:
    """
    Function to add tracking parameters, metrics and artifacts
    :param p1: user input one
    :param p2: uder input two
    :return: None
    """

    # set the tracking uri
    mlflow.set_tracking_uri('http://localhost:5000')

    # create demo experiment
    # mlflow.create_experiment('Demo_Experiment')

    # Start run
    with mlflow.start_run():
        mlflow.set_tag('Version', 'V1.0.0')  # set up the tag with version V1.0.0
        # log the input paramteres
        mlflow.log_param("param1", p1)
        mlflow.log_param("param2", p2)

        # log metric
        metric = evalution_metric(p1, p2)
        mlflow.log_metric('Eval_metric', metric)

        # create and log artifact
        os.makedirs('Execution_Time', exist_ok=True)
        with open('Execution_Time/time_logs.txt', 'wt') as f:
            f.write(f"Artifact created at: {time.asctime()}")
        mlflow.log_artifact("Execution_Time")

    return None


def evalution_metric(p1: int, p2: int) -> int:
    """
    Function to add squre of both p1 and p2 as dummy evalution metric
    :param p1: user input param1
    :param p2: user input param2
    :return: evaluation metric
    """
    metric = p1 ** 2 + p2 ** 2

    return metric


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--param1", type=int, default=5)
    args.add_argument("--param2", type=int, default=5)
    parsed_args = args.parse_args()

    # execute tracking function
    mlflow_tracking(parsed_args.param1,parsed_args.param2)

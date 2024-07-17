#!/bin/bash

function parse_arguments() {
    BASE_ARGS=""
    DQN_ARGS=""
    HP_SEARCH_ARGS=""
    ADADQN_ARGS=""
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e | --experiment_name)
                EXPERIMENT_NAME=$2
                shift
                shift
                ;;
            -frs | --first_seed)
                FIRST_SEED=$2
                shift
                shift
                ;;
            -lrs | --last_seed)
                LAST_SEED=$2
                shift
                shift
                ;;
            -rb | --replay_capacity)
                BASE_ARGS="$BASE_ARGS -rb $2"
                shift
                shift
                ;;
            -bs | --batch_size)
                BASE_ARGS="$BASE_ARGS -bs $2"
                shift
                shift
                ;;
            -n | --update_horizon)
                BASE_ARGS="$BASE_ARGS -n $2"
                shift
                shift
                ;;
            -gamma | --gamma)
                BASE_ARGS="$BASE_ARGS -gamma $2"
                shift
                shift
                ;;
            -hor | --horizon)
                BASE_ARGS="$BASE_ARGS -hor $2"
                shift
                shift
                ;;
            -utd | --update_to_data)
                BASE_ARGS="$BASE_ARGS -utd $2"
                shift
                shift
                ;;
            -tuf | --target_update_frequency)
                BASE_ARGS="$BASE_ARGS -tuf $2"
                shift
                shift
                ;;
            -n_init | --n_initial_samples)
                BASE_ARGS="$BASE_ARGS -n_init $2"
                shift
                shift
                ;;
            -eps_e | --end_epsilon)
                BASE_ARGS="$BASE_ARGS -eps_e $2"
                shift
                shift
                ;;
            -eps_dur | --duration_epsilon)
                BASE_ARGS="$BASE_ARGS -eps_dur $2"
                shift
                shift
                ;;
            -ne | --n_epochs)
                BASE_ARGS="$BASE_ARGS -ne $2"
                shift
                shift
                ;;
            -spe | --n_training_steps_per_epoch)
                BASE_ARGS="$BASE_ARGS -spe $2"
                shift
                shift
                ;;
            -g | --gpu)
                GPU=true
                shift
                ;;
            # DQN Specific
            -hl | --hidden_layers)
                shift
                HIDDEN_LAYER=""
                # parse all the layers till next flag encountered
                while [[ $1 != -* && $# -gt 0 ]]; do
                    HIDDEN_LAYER="$HIDDEN_LAYER $1"
                    shift
                done
                DQN_ARGS="$DQN_ARGS -hl $HIDDEN_LAYER"
                ;;
            -a | --activation)
                shift
                ACTIVATION=""
                # parse all the layers till next flag encountered
                while [[ $1 != -* && $# -gt 0 ]]; do
                    ACTIVATION="$ACTIVATION $1"
                    shift
                done
                DQN_ARGS="$DQN_ARGS -a $ACTIVATION"
                ;;
            -lr | --lr)
                DQN_ARGS="$DQN_ARGS -lr $2"
                shift
                shift
                ;;
            -o | --optimizer)
                DQN_ARGS="$DQN_ARGS -o $2"
                shift
                shift
                ;;
            -l | --loss)
                DQN_ARGS="$DQN_ARGS -l $2"
                shift
                shift
                ;;
            # Hyperparameter search Specific
            -nlr | --n_layers_range)
                HP_SEARCH_ARGS="$HP_SEARCH_ARGS -nlr $2 $3"
                shift
                shift
                shift
                ;;
            -nnr | --n_neurons_range)
                HP_SEARCH_ARGS="$HP_SEARCH_ARGS -nnr $2 $3"
                shift
                shift
                shift
                ;;
            -as | --activations)
                shift
                ACTIVATIONS=""
                # parse all the layers till next flag encountered
                while [[ $1 != -* && $# -gt 0 ]]; do
                    ACTIVATIONS="$ACTIVATIONS $1"
                    shift
                done
                HP_SEARCH_ARGS="$HP_SEARCH_ARGS -as $ACTIVATIONS"
                ;;
            -lrr | --lr_range)
                HP_SEARCH_ARGS="$HP_SEARCH_ARGS -lrr $2 $3"
                shift
                shift
                shift
                ;;
            -os | --optimizers)
                shift
                OPTIMIZERS=""
                # parse all the layers till next flag encountered
                while [[ $1 != -* && $# -gt 0 ]]; do
                    OPTIMIZERS="$OPTIMIZERS $1"
                    shift
                done
                HP_SEARCH_ARGS="$HP_SEARCH_ARGS -os $OPTIMIZERS"
                ;;
            -ls | --losses)
                shift
                LOSSES=""
                # parse all the layers till next flag encountered
                while [[ $1 != -* && $# -gt 0 ]]; do
                    LOSSES="$LOSSES $1"
                    shift
                done
                HP_SEARCH_ARGS="$HP_SEARCH_ARGS -ls $LOSSES"
                ;;
            # AdaDQN Specific
            -nn | --n_networks)
                ADADQN_ARGS="$ADADQN_ARGS -rb $2"
                shift
                shift
                ;;
            -eoe | --end_online_exp)
                ADADQN_ARGS="$ADADQN_ARGS -rb $2"
                shift
                shift
                ;;
            -?*)
                printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
                shift
                shift
                ;;
            ?*)
                printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
                shift
                ;;
        esac
    done

    if [[ $EXPERIMENT_NAME == "" ]]
    then
        echo "experiment name is missing, use -e" >&2
        exit
    elif ( [[ $FIRST_SEED = "" ]] || [[ $LAST_SEED = "" ]] )
    then
        echo "you need to specify -frs and -lrs" >&2
        exit
    fi
    if [[ $GPU == "" ]]
    then
        GPU=false
    fi
}
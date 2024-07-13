#!/bin/bash

function parse_arguments() {
    BASE_ARGS=""
    DQN_ARGS=""
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
                RB_CAPACITY=$2
                BASE_ARGS="$BASE_ARGS -rb $RB_CAPACITY"
                shift
                shift
                ;;
            -B | --batch_size)
                BATCH_SIZE=$2
                BASE_ARGS="$BASE_ARGS -B $BATCH_SIZE"
                shift
                shift
                ;;
            -n | --update_horizon)
                UPDATE_HORIZON=$2
                BASE_ARGS="$BASE_ARGS -n $UPDATE_HORIZON"
                shift
                shift
                ;;
            -gamma | --gamma)
                GAMMA=$2
                BASE_ARGS="$BASE_ARGS -gamma $GAMMA"
                shift
                shift
                ;;
            -lr | --lr)
                LEARNING_RATE=$2
                BASE_ARGS="$BASE_ARGS -lr $LEARNING_RATE"
                shift
                shift
                ;;
            -H | --horizon)
                HORIZON=$2
                BASE_ARGS="$BASE_ARGS -H $HORIZON"
                shift
                shift
                ;;
            -utd | --update_to_data)
                UPDATE_TO_DATA=$2
                BASE_ARGS="$BASE_ARGS -utd $UPDATE_TO_DATA"
                shift
                shift
                ;;
            -T | --target_update_period)
                TARGET_UPDATE_PERIOD=$2
                BASE_ARGS="$BASE_ARGS -T $TARGET_UPDATE_PERIOD"
                shift
                shift
                ;;
            -n_init | --n_initial_samples)
                N_INITIAL_SAMPLES=$2
                BASE_ARGS="$BASE_ARGS -n_init $N_INITIAL_SAMPLES"
                shift
                shift
                ;;
            -eps_e | --end_epsilon)
                END_EPSILON=$2
                BASE_ARGS="$BASE_ARGS -eps_e $END_EPSILON"
                shift
                shift
                ;;
            -eps_dur | --duration_epsilon)
                DURATION_EPSILON=$2
                BASE_ARGS="$BASE_ARGS -eps_dur $DURATION_EPSILON"
                shift
                shift
                ;;
            -E | --n_epochs)
                N_EPOCHS=$2
                BASE_ARGS="$BASE_ARGS -E $N_EPOCHS"
                shift
                shift
                ;;
            -spe | --n_training_steps_per_epoch)
                N_TRAINING_STEPS_PER_EPOCH=$2
                BASE_ARGS="$BASE_ARGS -spe $N_TRAINING_STEPS_PER_EPOCH"
                shift
                shift
                ;;
            -g | --gpu)
                GPU=true
                shift
                ;;
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
            -nn | --n_networks)
                ADADQN_ARGS="$ADADQN_ARGS -rb $2"
                shift
                shift
                ;;
            -nlr | --n_layers_range)
                ADADQN_ARGS="$ADADQN_ARGS -nlr $2 $3"
                shift
                shift
                shift
                ;;
            -nnr | --n_neurons_range)
                ADADQN_ARGS="$ADADQN_ARGS -nnr $2 $3"
                shift
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
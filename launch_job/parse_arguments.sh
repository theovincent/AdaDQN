#!/bin/bash

function parse_arguments() {
    # Get ALGO_NAME from the last word of the file name
    IFS='_' read -ra splitted_file_name <<< $(basename $0)
    ALGO_NAME=${splitted_file_name[-1]::-3}
    ENV_NAME=$(basename $(dirname ${0}))

    ARGS=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --algo_name)
                ALGO_NAME=$2
                shift
                shift
                ;;
            --env_name)
                ENV_NAME=$2
                shift
                shift
                ;;
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
            -nps | --n_parallel_seeds)
                N_PARALLEL_SEEDS=$2
                shift
                shift
                ;;
            -rb | --replay_capacity)
                ARGS="$ARGS -rb $2"
                shift
                shift
                ;;
            -bs | --batch_size)
                ARGS="$ARGS -bs $2"
                shift
                shift
                ;;
            -n | --update_horizon)
                ARGS="$ARGS -n $2"
                shift
                shift
                ;;
            -gamma | --gamma)
                ARGS="$ARGS -gamma $2"
                shift
                shift
                ;;
            -hor | --horizon)
                ARGS="$ARGS -hor $2"
                shift
                shift
                ;;
            -utd | --update_to_data)
                ARGS="$ARGS -utd $2"
                shift
                shift
                ;;
            -tuf | --target_update_frequency)
                ARGS="$ARGS -tuf $2"
                shift
                shift
                ;;
            -nis | --n_initial_samples)
                ARGS="$ARGS -nis $2"
                shift
                shift
                ;;
            -eps_e | --end_epsilon)
                ARGS="$ARGS -eps_e $2"
                shift
                shift
                ;;
            -eps_dur | --duration_epsilon)
                ARGS="$ARGS -eps_dur $2"
                shift
                shift
                ;;
            -ne | --n_epochs)
                ARGS="$ARGS -ne $2"
                shift
                shift
                ;;
            -spe | --n_training_steps_per_epoch)
                ARGS="$ARGS -spe $2"
                shift
                shift
                ;;
            -g | --gpu)
                GPU=true
                shift
                ;;
            # dqn specific
            -o | --optimizer)
                ARGS="$ARGS -o $2"
                shift
                shift
                ;;
            -lr | --learning_rate)
                ARGS="$ARGS -lr $2"
                shift
                shift
                ;;
            -l | --loss)
                ARGS="$ARGS -l $2"
                shift
                shift
                ;;
            -fs | --features)
                shift
                FEATURES=""
                # parse all the layers till next flag encountered
                while [[ $1 != -* && $# -gt 0 ]]; do
                    FEATURES="$FEATURES $1"
                    shift
                done
                ARGS="$ARGS -fs $FEATURES"
                ;;
            -as | --activations)
                shift
                ACTIVATIONS=""
                # parse all the layers till next flag encountered
                while [[ $1 != -* && $# -gt 0 ]]; do
                    ACTIVATIONS="$ACTIVATIONS $1"
                    shift
                done
                ARGS="$ARGS -as $ACTIVATIONS"
                ;;
            # adadqnstatic specific
            -nn | --n_networks)
                ARGS="$ARGS -nn $2"
                shift
                shift
                ;;
            -osl | --optimizers_list)
                shift
                OPTIMIZERS_LIST=""
                # parse all the layers till next flag encountered
                while [[ $1 != -* && $# -gt 0 ]]; do
                    OPTIMIZERS_LIST="$OPTIMIZERS_LIST $1"
                    shift
                done
                ARGS="$ARGS -osl $OPTIMIZERS_LIST"
                ;;
            -lrsl | --learning_rates_list)
                shift
                LEARNING_RATES_LIST=""
                # parse all the layers till next flag encountered
                while [[ $1 != -* && $# -gt 0 ]]; do
                    LEARNING_RATES_LIST="$LEARNING_RATES_LIST $1"
                    shift
                done
                ARGS="$ARGS -lrsl $LEARNING_RATES_LIST"
                ;;
            -lsl | --losses_list)
                shift
                LOSSES_LIST=""
                # parse all the layers till next flag encountered
                while [[ $1 != -* && $# -gt 0 ]]; do
                    LOSSES_LIST="$LOSSES_LIST $1"
                    shift
                done
                ARGS="$ARGS -lsl $LOSSES_LIST"
                ;;
            -fsl | --features_list)
                shift
                FEATURES_LIST=""
                # parse all the layers till next flag encountered
                while [[ $1 != -* && $# -gt 0 ]]; do
                    FEATURES_LIST="$FEATURES_LIST $1"
                    shift
                done
                ARGS="$ARGS -fsl $FEATURES_LIST"
                ;;
            -asl | --activations_list)
                shift
                ACTIVATIONS_LIST=""
                # parse all the layers till next flag encountered
                while [[ $1 != -* && $# -gt 0 ]]; do
                    ACTIVATIONS_LIST="$ACTIVATIONS_LIST $1"
                    shift
                done
                ARGS="$ARGS -asl $ACTIVATIONS_LIST"
                ;;
            -eoe | --end_online_exp)
                ARGS="$ARGS -eoe $2"
                shift
                shift
                ;;
            # Hyperparameter search specific
            -os | --optimizers)
                shift
                OPTIMIZERS=""
                # parse all the layers till next flag encountered
                while [[ $1 != -* && $# -gt 0 ]]; do
                    OPTIMIZERS="$OPTIMIZERS $1"
                    shift
                done
                ARGS="$ARGS -os $OPTIMIZERS"
                ;;
            -lrr | --learning_rate_range)
                ARGS="$ARGS -lrr $2 $3"
                shift
                shift
                shift
                ;;
            -ls | --losses)
                shift
                LOSSES=""
                # parse all the layers till next flag encountered
                while [[ $1 != -* && $# -gt 0 ]]; do
                    LOSSES="$LOSSES $1"
                    shift
                done
                ARGS="$ARGS -ls $LOSSES"
                ;;
            -nlr | --n_layers_range)
                ARGS="$ARGS -nlr $2 $3"
                shift
                shift
                shift
                ;;
            -nnr | --n_neurons_range)
                ARGS="$ARGS -nnr $2 $3"
                shift
                shift
                shift
                ;;
            # adadqn specific
            -ocp | --optimizer_change_probability)
                ARGS="$ARGS -ocp $2"
                shift
                shift
                ;;
            -acp | --architecture_change_probability)
                ARGS="$ARGS -acp $2"
                shift
                shift
                ;;
            # rsdqn specific
            -nephp | --n_epochs_per_hyperparameter)
                ARGS="$ARGS -nephp $2"
                shift
                shift
                ;;
            # dehbdqn specific
            -minnephp | --min_n_epochs_per_hyperparameter)
                ARGS="$ARGS -minnephp $2"
                shift
                shift
                ;;
            -maxnephp | --max_n_epochs_per_hyperparameter)
                ARGS="$ARGS -maxnephp $2"
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

    [ -d experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME ] || mkdir -p experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME
}
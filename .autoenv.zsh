__conda_setup="$("${XDG_PREFIX_HOME}/miniconda3/bin/conda" 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "${XDG_PREFIX_HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
        . "${XDG_PREFIX_HOME}/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="${XDG_PREFIX_HOME}/miniconda3/bin:$PATH"
    fi
fi
conda_env_name=vi-depth
conda activate ${conda_env_name}
export CUDA_HOME=${CONDA_PREFIX}

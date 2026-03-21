# bash snippet for adding Compass-specific environment variables

if [ "${COMPASS_MACHINE:-}" = "chicoma-cpu" ] || \
   [ "${COMPASS_MACHINE:-}" = "pm-cpu" ] || \
   [ "${COMPASS_MACHINE:-}" = "pm-gpu" ]; then
    export NETCDF="${CRAY_NETCDF_HDF5PARALLEL_PREFIX}"
    export NETCDFF="${CRAY_NETCDF_HDF5PARALLEL_PREFIX}"
    export PNETCDF="${CRAY_PARALLEL_NETCDF_PREFIX}"
else
    export NETCDF="$(dirname "$(dirname "$(which nc-config)")")"
    export NETCDFF="$(dirname "$(dirname "$(which nf-config)")")"
    export PNETCDF="$(dirname "$(dirname "$(which pnetcdf-config)")")"
fi

if [ "${COMPASS_MPI:-}" = "mvapich" ]; then
    export MV2_ENABLE_AFFINITY=0
    export MV2_SHOW_CPU_BINDING=1
fi

if [ -n "${MACHE_DEPLOY_SPACK_LIBRARY_VIEW:-}" ]; then
    export PIO="${MACHE_DEPLOY_SPACK_LIBRARY_VIEW}"
    export METIS_ROOT="${MACHE_DEPLOY_SPACK_LIBRARY_VIEW}"
    export PARMETIS_ROOT="${MACHE_DEPLOY_SPACK_LIBRARY_VIEW}"
else
    export PIO="${CONDA_PREFIX}"
    export OPENMP_INCLUDE="-I${CONDA_PREFIX}/include"
fi

export HAVE_ADIOS=false

albany_flag_file="${MACHE_DEPLOY_SPACK_LIBRARY_VIEW:-}/export_albany.in"
if [ -f "${albany_flag_file}" ]; then
    # shellcheck source=/dev/null
    source "${albany_flag_file}"
    export ALBANY_LINK_LIBS

    stdcxx="-lstdc++"
    if [ "$(uname)" = "Darwin" ]; then
        stdcxx="-lc++"
    fi

    mpicxx=""
    if [ "${COMPASS_MPI:-}" = "openmpi" ] && \
       { [ "${COMPASS_MACHINE:-}" = "anvil" ] || \
         [ "${COMPASS_MACHINE:-}" = "chrysalis" ]; }; then
        mpicxx="-lmpi_cxx"
    fi

    export MPAS_EXTERNAL_LIBS="${MPAS_EXTERNAL_LIBS} ${ALBANY_LINK_LIBS} ${stdcxx} ${mpicxx}"
fi

export USE_PIO2=true
export OPENMP=true
export HDF5_USE_FILE_LOCKING=FALSE

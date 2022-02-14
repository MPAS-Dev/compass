<streams>

<immutable_stream name="restart"
                  filename_template="rst.$Y_$M_$D.nc"
                  filename_interval="output_interval"
                  reference_time="2007-07-01_00:00:00"
                  output_interval="0001-00-00_00:00:00"/>

<stream name="output"
        output_interval="0000-00-01_00:00:00"
        reference_time="0000-01-01_00:00:00"
        clobber_mode="overwrite">
</stream>

</streams>

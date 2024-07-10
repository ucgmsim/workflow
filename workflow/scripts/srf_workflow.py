#!/usr/bin/env python3
import logging
import os
from argparse import ArgumentParser
from pathlib import Path

from workflow.scripts import generate_srf

logging.basicConfig(level=logging.DEBUG)

# --- Import Pegasus API ------------------------------------------------------
from Pegasus.api import (
    Directory,
    File,
    FileServer,
    Job,
    Namespace,
    Operation,
    Properties,
    Site,
    SiteCatalog,
    Transformation,
    TransformationCatalog,
    Workflow,
)


class ProcessWorkflow:
    wf = None
    sc = None
    tc = None
    props = None

    dagfile = None
    wf_name = None
    wf_dir = None

    # --- Init ----------------------------------------------------------------
    def __init__(self, dagfile="workflow.yml"):
        self.dagfile = dagfile
        self.wf_name = "process"
        self.wf_dir = str(Path(__file__).parent.resolve())

    # --- Write files in directory --------------------------------------------
    def write(self):
        if not self.sc is None:
            self.sc.write()
        self.props.write()
        self.tc.write()
        self.wf.write()

    # --- Configuration (Pegasus Properties) ----------------------------------
    def create_pegasus_properties(self):
        self.props = Properties()

    # --- Site Catalog --------------------------------------------------------
    def create_sites_catalog(self, exec_site_name="condorpool"):
        self.sc = SiteCatalog()

        shared_scratch_dir = os.path.join(self.wf_dir, "scratch")
        local_storage_dir = os.path.join(self.wf_dir, "output")

        local = Site("local").add_directories(
            Directory(Directory.SHARED_SCRATCH, shared_scratch_dir).add_file_servers(
                FileServer("file://" + shared_scratch_dir, Operation.ALL)
            ),
            Directory(Directory.LOCAL_STORAGE, local_storage_dir).add_file_servers(
                FileServer("file://" + local_storage_dir, Operation.ALL)
            ),
        )

        exec_site = (
            Site(exec_site_name)
            .add_pegasus_profile(style="condor")
            .add_condor_profile(universe="vanilla")
            .add_profiles(Namespace.PEGASUS, key="data.configuration", value="condorio")
        )

        self.sc.add_sites(local, exec_site)

    # --- Transformation Catalog (Executables and Containers) -----------------
    def create_transformation_catalog(self, exec_site_name="condorpool"):
        self.tc = TransformationCatalog()

        # Add the ls executable
        generate_srf_script = Transformation(
            "generate_srf",
            site=exec_site_name,
            pfn=generate_srf.__file__,
            is_stageable=False,
        )

        self.tc.add_transformations(generate_srf_script)

    # --- Create Workflow -----------------------------------------------------
    def create_workflow(self):
        self.wf = Workflow(self.wf_name, infer_dependencies=True)

        # Add a ls job
        realisation = File("realisation.json")
        realisation_srf = File("realisation.srf")
        srf_generation_job = (
            Job("generate_srf")
            .add_args(realisation, realisation_srf)
            .add_inputs(realisation)
            .add_outputs(realisation_srf)
        )

        self.wf.add_jobs(srf_generation_job)


def main():
    parser = ArgumentParser(description="Pegasus Process Workflow")

    parser.add_argument(
        "-s",
        "--skip_sites_catalog",
        action="store_true",
        help="Skip site catalog creation",
    )
    parser.add_argument(
        "-e",
        "--execution_site_name",
        metavar="STR",
        type=str,
        default="condorpool",
        help="Execution site name (default: condorpool)",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="STR",
        type=str,
        default="workflow.yml",
        help="Output file (default: workflow.yml)",
    )

    args = parser.parse_args()

    workflow = ProcessWorkflow(args.output)

    if not args.skip_sites_catalog:
        print("Creating execution sites...")
        workflow.create_sites_catalog(args.execution_site_name)

    print("Creating workflow properties...")
    workflow.create_pegasus_properties()

    print("Creating transformation catalog...")
    workflow.create_transformation_catalog(args.execution_site_name)

    print("Creating process workflow dag...")
    workflow.create_workflow()

    workflow.write()


if __name__ == "__main__":
    main()

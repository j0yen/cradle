//! cradle CLI entry point.

#![forbid(unsafe_code)]
#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::ignored_unit_patterns,
    clippy::needless_pass_by_value
)]

use std::process::ExitCode;

use clap::Parser;

use cradle::cli::{
    BakeArgs, BuildArgs, Cli, Command, HarvestArgs, StatusArgs, TrainArgs,
};
use cradle::orchestrator::{
    OrchestrationError, collect_statuses, run_bake, run_harvest, run_train,
};

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Harvest(args) => handle_result(do_harvest(args)),
        Command::Train(args) => handle_result(do_train(args)),
        Command::Bake(args) => handle_result(do_bake(args)),
        Command::Build(args) => handle_result(do_build(args)),
        Command::Status(args) => handle_result(do_status(args)),
    }
}

fn handle_result(result: Result<(), OrchestrationError>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("cradle: {e}");
            ExitCode::from(1)
        }
    }
}

fn do_harvest(args: HarvestArgs) -> Result<(), OrchestrationError> {
    let result = run_harvest(
        &args.models_dir,
        &args.model,
        args.transcripts_dir.as_deref(),
        args.out_dir.as_deref(),
    )?;
    eprintln!("{}", result.stats.one_line());
    println!(
        "cradle: harvest {} -> {}",
        result.model,
        result.data_dir.display()
    );
    Ok(())
}

fn do_train(args: TrainArgs) -> Result<(), OrchestrationError> {
    run_train(&args.models_dir, &args.model, &args.runner)?;
    println!("cradle: train {} (via {}) ok", args.model, args.runner);
    Ok(())
}

fn do_bake(args: BakeArgs) -> Result<(), OrchestrationError> {
    // Intentionally returns BakeDeferred — see AC8.
    run_bake(&args.model)
}

fn do_build(args: BuildArgs) -> Result<(), OrchestrationError> {
    let result = run_harvest(
        &args.models_dir,
        &args.model,
        args.transcripts_dir.as_deref(),
        None,
    )?;
    eprintln!("{}", result.stats.one_line());
    println!("cradle: build {} harvest ok", args.model);
    if !args.skip_train {
        match run_train(&args.models_dir, &args.model, "uv") {
            Ok(()) => println!("cradle: build {} train ok", args.model),
            Err(e) => {
                eprintln!("cradle: build {} train failed: {e}", args.model);
                return Err(e);
            }
        }
    }
    println!("cradle: build {}: phase 2 deferred (see PRD-cradle-bake-integration.md)", args.model);
    Ok(())
}

fn do_status(args: StatusArgs) -> Result<(), OrchestrationError> {
    let statuses = collect_statuses(&args.models_dir)?;
    if args.json {
        let payload = serde_json::json!({
            "schema": "cradle.status.v1",
            "models": statuses,
        });
        let line = serde_json::to_string_pretty(&payload).map_err(|e| {
            OrchestrationError::Io(std::io::Error::other(e))
        })?;
        println!("{line}");
    } else if statuses.is_empty() {
        println!("cradle: no models found");
    } else {
        for s in &statuses {
            println!("{}", s.one_line());
        }
    }
    Ok(())
}

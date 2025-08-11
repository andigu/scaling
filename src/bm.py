from simulate import Algorithm, SurfaceCode, NoiseModel, CodeFactory, StimSimulator
import stim
import multiprocessing
import sinter
import numpy as np
import json
import pandas as pd
import click
from beliefmatching import BeliefMatchingSinterDecoder

def generate_example_tasks():
    p = 2.1
    for d in range(3, 20, 2):
        alg = Algorithm.build_memory(cycles=d)
        cf = CodeFactory(SurfaceCode, {'d': d})
        noise_model = NoiseModel.get_scaled_noise_model(p).without_loss()
        sim = StimSimulator(alg, noise_model, cf)
        dummy_circuit = sim.stim_circ
        data_indices = np.arange(d**2).reshape((d,d))
        zlog = data_indices[:,d//2]-d**2
        xlog = data_indices[d//2,:]-d**2
        dummy_circuit.append_from_stim_program_text('OBSERVABLE_INCLUDE(0) ' + ' '.join([f"rec[{i}]" for i in zlog]))

        yield sinter.Task(
            circuit=dummy_circuit,
            json_metadata={
            'd': d
            },
        )

def main():
    # Collect the samples (takes a few minutes).
    samples = sinter.collect(
        num_workers=multiprocessing.cpu_count()-1,
        max_shots=50_000,
        tasks=generate_example_tasks(),
        decoders=['beliefmatching'],
        print_progress=True,
        custom_decoders={'beliefmatching': BeliefMatchingSinterDecoder()},
        save_resume_filepath='bm-sinter.csv'
    )

    # Print samples as CSV data.
    with open(f'bm_samples.csv', 'w+') as f:
        f.write(sinter.CSV_HEADER + '\n')
        for sample in samples:
            f.write(sample.to_csv_line() + '\n')
    df = pd.read_csv(f"bm_samples.csv", skipinitialspace=True)
    rounds = [json.loads(x)['rounds'] for x in df["json_metadata"]]
    df['d'] = d
    df['error_rate'] = df['errors']/df['shots']
    df['std_error_rate'] = np.sqrt(df['error_rate']*(1-df['error_rate'])/df['shots'])
    df = df.sort_values(by='d')
    df.to_csv(f'bm_samples.csv', index=False)
    
if __name__ == "__main__":
    main()

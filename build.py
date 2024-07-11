import shutil
import os
from general import relative_path


def main() -> None:
    for file in os.listdir(relative_path('build_artifacts')):
        if not file.startswith('.'):
            shutil.copy(relative_path(f"build_artifacts/{file}"), relative_path(f"build/{file}"))
    vision_model = None
    for file in sorted(os.listdir(relative_path('models')), reverse=True):
        if file.endswith('.pt'):
            vision_model = file
            break
    if vision_model is None:
        raise Exception('vision model not found')
    print(f'{vision_model=}')
    shutil.copy(relative_path(f"models/{vision_model}"), relative_path('build/vision_model.pt'))
    shutil.copy(relative_path('.env'), relative_path('build/.env'))
    shutil.copy(relative_path('llm_models/unsloth.Q4_K_M.gguf'), relative_path('build/llm.Q4_K_M.gguf'))
    shutil.copy(relative_path('data/ingredients/lists/animal.json'), relative_path('build/animal.json'))
    shutil.copy(relative_path('data/ingredients/lists/sometimes_animal.json'),
                relative_path('build/sometimes_animal.json'))


if __name__ == '__main__':
    main()

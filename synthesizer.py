class Synthesizer:
    def __init__(self, model_path, config_path) -> None:
        self.__model_path = model_path
        self.__config_path = config_path
        self.__counter = 0
        self.__last = ''

    def generate_voice(self, text, output_path, index = None):
        out_path = f'{output_path}/{f"{self.__counter}.wav" if index == None else index}'
        subprocess.run([
            'tts',
            '--text', text,
            '--model_path', self.__model_path,
            '--config_path', self.__config_path,
            '--out_path', out_path
        ])
        self.__last = out_path
        if self.__counter % 5 == 0: print(f'{self.__counter} files generated at {output_path}')
        self.__counter += 1

    def generate_dubbed(self, segments, output_path):
      audio = AudioSegment.empty()
      for i in range(len(segments)):
        segment = segments[i]
        filename = f"segment-{i}.wav"
        self.generate_voice(segment['translation'], './output', filename)
        audio += AudioSegment.from_file(f"./output/{filename}")
        if i < len(segments) - 1 and (segments[i + 1]['start'] - segments[i]['end']) > 0:
          audio += AudioSegment.silent(duration = (1000 * (segments[i + 1]['start'] - segments[i]['end'])))
      audio.export(output_path, format="wav")
      self.__last = output_path

    def play(self):
      if self.__last == '': return
      sound_file = self.__last
      display(Audio(sound_file, autoplay=True))
    
if __name__ == '__main__':
   model_path = input('model path:')
   config_path = input('config path')
   tts = Synthesizer(model_path, config_path)
   tts.generate_voice(input('text in farsi:'), '.', 'out.wav')
   print('finished!')
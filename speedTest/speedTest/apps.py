from django.apps import AppConfig

class SampleappConfig(AppConfig):
    name = 'sampleApp'

class SpeedTest(AppConfig):
    app_label='speedTest'
    name= "speedTest"
    def __str__(self):
        return self.name + ": " + str(self.app_label)
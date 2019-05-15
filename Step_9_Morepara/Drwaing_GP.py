import matplotlib.pyplot as plt
import pandas as pd

VER_NAME = 'VER_11_LSTM'
NB = 5601

class GP:
    def __init__(self):
        self.gp_db = pd.read_csv('{}/log/{}.csv'.format(VER_NAME, NB))
        self.fig = plt.figure(constrained_layout=True, figsize=(14, 10)) # constrained_layout=True,
        self.gs = self.fig.add_gridspec(14, 3)
        self.axs = [self.fig.add_subplot(self.gs[0:3, :]),
                    self.fig.add_subplot(self.gs[3:6, :]),
                    self.fig.add_subplot(self.gs[6:7, :]),
                    self.fig.add_subplot(self.gs[7:8, :]),
                    self.fig.add_subplot(self.gs[8:9, :]),
                    self.fig.add_subplot(self.gs[9:14, :]),
                    ]

    def make_gp(self):
        self.axs[0].plot(self.gp_db['time'], self.gp_db['Reactor_power'], 'g', label='Power')
        self.axs[0].plot(self.gp_db['time'], self.gp_db['Reactor_power_up'], 'b', label='Power_hi_bound')
        self.axs[0].plot(self.gp_db['time'], self.gp_db['Reactor_power_low'], 'r', label='Power_low_bound')
        self.axs[0].legend(loc=2, fontsize=9)
        self.axs[0].set_ylabel('Reactor Power [%]')
        # self.axs[0].set_title('Reactor Power [%]', fontsize=10)
        self.axs[0].grid()
        #
        self.axs[1].plot(self.gp_db['time'], self.gp_db['Mwe'], 'g', label='Mwe')
        self.axs[1].plot(self.gp_db['time'], self.gp_db['Mwe_up'], 'b', label='Mwe_hi_bound')
        self.axs[1].plot(self.gp_db['time'], self.gp_db['Mwe_low'], 'r', label='Mwe_low_bound')
        self.axs[1].legend(loc=2, fontsize=9)
        self.axs[1].set_ylabel('Electrical Power [MWe]')
        # self.axs[1].set_title('Electrical Power [MWe]', fontsize=10)
        self.axs[1].grid()
        #
        self.axs[2].plot(self.gp_db['time'], self.gp_db['Turbine_set'], 'r')
        self.axs[2].plot(self.gp_db['time'], self.gp_db['Turbine_real'], 'b')
        self.axs[2].set_yticks((900, 1800))
        self.axs[2].set_yticklabels(('900', '1800'))
        self.axs[2].set_ylabel('Turbine RPM')
        # self.axs[2].set_title('Turbine RPM', fontsize=10)
        self.axs[2].grid()
        #
        self.axs[3].plot(self.gp_db['time'], self.gp_db['Rod_A'], 'black')
        self.axs[3].set_yticks((-1, 0, 1))
        self.axs[3].set_yticklabels(('In', 'Stay', 'Out'))
        # self.axs[3].set_ylabel('Rod Control')
        self.axs[3].set_title('Rod Control', fontsize=10)
        self.axs[3].grid()
        #
        self.axs[4].plot(self.gp_db['time'], self.gp_db['Tur_A'], 'black')
        self.axs[4].set_yticks((-1, 0, 1))
        self.axs[4].set_yticklabels(('Down', 'Stay', 'Up'))
        # self.axs[4].set_ylabel('Load Rate Control')
        self.axs[4].set_title('Load Rate Control', fontsize=10)
        self.axs[4].grid()
        #
        self.axs[5].plot(self.gp_db['time'], self.gp_db['Net_break'], label='Net break')
        self.axs[5].plot(self.gp_db['time'], self.gp_db['Trip_block'], label='Trip block')
        self.axs[5].plot(self.gp_db['time'], self.gp_db['Stem_pump'], label='Stem dump valve auto')
        self.axs[5].plot(self.gp_db['time'], self.gp_db['Heat_pump'], label='Heat pump')
        self.axs[5].plot(self.gp_db['time'], self.gp_db['MF1'], label='Main Feed Water Pump 1')
        self.axs[5].plot(self.gp_db['time'], self.gp_db['MF2'], label='Main Feed Water Pump 2')
        self.axs[5].plot(self.gp_db['time'], self.gp_db['MF3'], label='Main Feed Water Pump 3')
        self.axs[5].plot(self.gp_db['time'], self.gp_db['CF1'], label='Condensor Pump 1')
        self.axs[5].plot(self.gp_db['time'], self.gp_db['CF2'], label='Condensor Pump 2')
        self.axs[5].plot(self.gp_db['time'], self.gp_db['CF3'], label='Condensor Pump 3')
        self.axs[5].set_yticks((0, 1))
        self.axs[5].set_yticklabels(('Off', 'On'))
        self.axs[5].set_xlabel('Time [s]')
        self.axs[5].legend(loc=2, fontsize=9)
        self.axs[5].grid()
        #
        self.fig.savefig(fname='gp.png', dpi=600, facecolor=None)
        plt.show()


if __name__ == '__main__':
    gp = GP()
    gp.make_gp()
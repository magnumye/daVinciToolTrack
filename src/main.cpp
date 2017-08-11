#include <QApplication>
#include "GraphicalUserInterface.h"


int main(int argc, char** argv)
{
    std::string config_path;

    config_path = "../config/sim_config_win.xml";   // Change here
    QApplication app(argc, argv);

    GraphicalUserInterface gui (config_path);
    gui.show();

    return app.exec();
}

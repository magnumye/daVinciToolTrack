#ifndef GraphicalUserInterface_H
#define GraphicalUserInterface_H

#include "ui_GraphicalUserInterface.h"

#include <vtkSmartPointer.h>

//#include <vtkOBBTree.h>
//#include <vtkAppendPolyData.h>



#include <QMainWindow>
#include <QThread>
#include <opencv2/opencv.hpp>

#include "Eigen/LU"

#include "simulator.h"

#include "mouse_interactor_style.h"

#include <mutex>

namespace Ui {
    class MainWindow;
}


class DeformationView;

class GraphicalUserInterface : public QMainWindow, private Ui::GraphicalUserInterface
{
    Q_OBJECT
public:

    GraphicalUserInterface(std::string config_path);

Q_SIGNALS:


public slots:

    virtual void slotExit();

    void updateRenderImage(QImage);

    void updateCameraImage(QImage);

	void setInfo1(QString message);

    void setInfo2(QString message);

protected:
    Simulator sim;

private slots:


    void on_loadButton_released();

    void on_live_checkBox_toggled(bool checked);

    void on_sim_checkBox_toggled(bool checked);

    void on_kine_checkBox_toggled(bool checked);

private:

    /**
     * Ray tracing
     * */
//    vtkSmartPointer<vtkCoordinate> m_coord;
//    vtkSmartPointer<customMouseInteractorStyle> m_style;

//    vtkSmartPointer<vtkOBBTree> m_tree;

//    vtkSmartPointer<vtkAppendPolyData> app_poly_filter;


};



#endif

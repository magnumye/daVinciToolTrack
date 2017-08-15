#include "GraphicalUserInterface.h"
#include <QPainter>
#include <QPen>

#include <vtkLight.h>
#include <vtkLightCollection.h>



// Constructor
GraphicalUserInterface::GraphicalUserInterface(std::string config_path) :
    sim(config_path)
{
    this->setupUi(this);
    sim.setupGUI(this);

	virtual_label->setVisible(false);

    // Set up action signals and slots
    connect(this->actionExit, SIGNAL(triggered()), this, SLOT(slotExit()));

    // Ray tracing
//    m_tree = vtkSmartPointer<vtkOBBTree>::New();
//    app_poly_filter = vtkSmartPointer<vtkAppendPolyData>::New();
}


void GraphicalUserInterface::slotExit()
{
    qApp->exit();
}

void GraphicalUserInterface::updateRenderImage(QImage qimage)
{
    if(virtual_label->isVisible()){
        virtual_label->setPixmap(QPixmap::fromImage(qimage));
        virtual_label->repaint();
    }
}

void GraphicalUserInterface::updateCameraImage(QImage qimage)
{
    if(camera_label->isVisible()){
        camera_label->setPixmap(QPixmap::fromImage(qimage));
        camera_label->repaint();
    }
}


void GraphicalUserInterface::on_loadButton_released()
{
    sim.init();
}

void GraphicalUserInterface::on_live_checkBox_toggled(bool checked)
{
    this->live_checkBox->setChecked(false);
}

void GraphicalUserInterface::on_sim_checkBox_toggled(bool checked)
{
	if (!checked)
	{
		virtual_label->setVisible(false);
	}
	else
	{
		virtual_label->setVisible(true);
	}
	sim.show_simulation(checked);
}

void GraphicalUserInterface::on_kine_checkBox_toggled(bool checked)
{
	sim.show_kinematics(checked);
}

void GraphicalUserInterface::setInfo1(QString message)
{
    info_label1->setText(message);
    info_label1->repaint();
}


void GraphicalUserInterface::setInfo2(QString message)
{
    info_label2->setText(message);
    info_label2->repaint();
}

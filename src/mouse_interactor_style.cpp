#include "mouse_interactor_style.h"
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSmartPointer.h>

vtkStandardNewMacro(customMouseInteractorStyle);

void customMouseInteractorStyle::OnLeftButtonDown()
{
  std::cout << "Pressed left mouse button." << std::endl;
  int x = this->Interactor->GetEventPosition()[0];
  int y = this->Interactor->GetEventPosition()[1];
  std::cout << "(x,y) = (" << x << "," << y << ")" << std::endl;
  vtkSmartPointer<vtkCoordinate> coordinate =
    vtkSmartPointer<vtkCoordinate>::New();
  coordinate->SetCoordinateSystemToDisplay();
  coordinate->SetValue(x,y,0);

  // This doesn't produce the right value if the sphere is zoomed in???
  double* world = coordinate->GetComputedWorldValue(this->Interactor->GetRenderWindow()->GetRenderers()->GetFirstRenderer());
  memcpy(select_3D, world, 3*sizeof(double));
  std::cout << "World coordinate: " << select_3D[0] << ", " << select_3D[1] << ", " << select_3D[2] << std::endl;


  // Forward events
  vtkInteractorStyleTrackballCamera::OnLeftButtonDown();
}

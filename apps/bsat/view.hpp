/*******************************************************************************
* Views abstract class
*
* Author: Levi Miller
* Created: 3/21/2021
* Checked: 2023-03-03
*******************************************************************************/

#ifndef _VIEW_H_
#define _VIEW_H_

#include "decs.hpp"

struct ImVec4;

class IView {
public:
    virtual const char* Name() = 0; // Name of the view

    virtual void Render() = 0; // Render the view
    virtual void Update() = 0; // Update the view, ex reload data after filter applied

    virtual bool IsUpdating() = 0; // Is the view updating

    virtual void PushTabStyle() = 0;
    virtual void PopTabStyle() = 0;

    // TODO: Remove these
    virtual void ClosePlot(Plot* plot) = 0; // Tell view to close a plot
    virtual void CloseWidget(Widget* widget) = 0; // Tell view to close a widget
    //virtual void PlotContextMenu() = 0; // Add to ImPlot context menu
    
    virtual void SessionAddFileCallback(BFile* file) = 0; // Called when a new file is added to the session

    // TODO: Remove, should pull from session
    virtual ImVec4 GetTabActiveColor() = 0; // How to color the tab

    virtual ~IView() { }

    //virtual const Session* GetSession() const = 0; // Get the session
    //virtual void SetSession(Session* session) = 0; // Set the session
};

#endif
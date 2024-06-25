#ifndef _FDECS_
#define _FDECS_

// TODO: This file is evil and I am working to be rid of it

// plot
class  Plot;
struct PlotField;
struct PlotFieldContainer;
struct PlotFieldRenderer;
struct PlotSettings;

// plotter
class Plotter;

class BFile;
class SignalFile;
class TBDFile;

// widgets
class Widget;
class Info_Widget;
class AnnotationDisplay_Widget;
class MeasurementDisplay_Widget;
class PlotControls_Widget;
class Fields_Widget;
class Colormap_Widget;
class Filter_Widget;

// tool
class Tool;

// session
class  Session;
struct Measurement;

// annotation
struct Annotation;

// color
class Colorizer;
class ColorizerSingle;
class ColorizerRange;
class ColorStack;

// filter
class TBDFilter;
class TBDIQFilter;
class FilterStack;

// analysis
class Analyzer;
class IQPulseStats;

#endif
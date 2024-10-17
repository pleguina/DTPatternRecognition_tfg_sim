//CMSSW script to print out the DT geometry information in a human-readable format and generate an XML file with the same information using Boost Property Tree.
//Pelayo Leguina 2024

#include "DataFormats/Math/interface/Rounding.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include <iostream>
#include <string>
#include <fstream>
#include <memory>
#include <sstream> // For string formatting

// Include Boost Property Tree headers
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

using namespace std;
using namespace cms;
using namespace edm;
using namespace cms_rounding;
using boost::property_tree::ptree;

class DTGeometryTest : public one::EDAnalyzer<> {
public:
  explicit DTGeometryTest(const ParameterSet&);

  void beginJob() override;
  void analyze(Event const& iEvent, EventSetup const&) override;
  void endJob() override;

private:
  const string m_label;
  const edm::ESGetToken<DTGeometry, MuonGeometryRecord> m_token;

  // Member variable for XML generation using Boost Property Tree
  ptree tree_;
  bool xmlWritten_;

  // Helper function to format GlobalVector
  std::string formatVector(const GlobalVector& vec) const;

  // Helper functions to format points
  std::string formatGlobalPoint(const GlobalPoint& point) const;
  std::string formatLocalPoint(const LocalPoint& point) const;
};

DTGeometryTest::DTGeometryTest(const ParameterSet& iConfig)
    : m_label(iConfig.getUntrackedParameter<string>("fromDataLabel", "")),
      m_token(esConsumes<DTGeometry, MuonGeometryRecord>(edm::ESInputTag{"", m_label})),
      xmlWritten_(false) // Initialize the tracking variable
{}

std::string DTGeometryTest::formatVector(const GlobalVector& vec) const {
  std::ostringstream oss;
  oss << "(" << vec.x() << ", " << vec.y() << ", " << vec.z() << ")";
  return oss.str();
}

std::string DTGeometryTest::formatGlobalPoint(const GlobalPoint& point) const {
  std::ostringstream oss;
  oss << "(" << point.x() << ", " << point.y() << ", " << point.z() << ")";
  return oss.str();
}

std::string DTGeometryTest::formatLocalPoint(const LocalPoint& point) const {
  std::ostringstream oss;
  oss << "(" << point.x() << ", " << point.y() << ", " << point.z() << ")";
  return oss.str();
}

void DTGeometryTest::beginJob() {
  // Initialize the XML tree with root element and its attributes
  tree_.put("<xmlattr>.version", "1.0");
  tree_.put("<xmlattr>.encoding", "UTF-8");
  tree_.add("DTGeometry", ""); // Adds the root node without assigning it to a variable
}

void DTGeometryTest::analyze(const Event&, const EventSetup& iEventSetup) {
  LogVerbatim("DTGeometryTest") << "DTGeometryTest::analyze: " << m_label;
  ESTransientHandle<DTGeometry> pDD = iEventSetup.getTransientHandle(m_token);

  LogVerbatim("DTGeometryTest") << " Geometry node for DTGeom is " << (pDD.isValid() ? "valid" : "not valid");
  LogVerbatim("DTGeometryTest") << " I have " << pDD->detTypes().size() << " detTypes";
  LogVerbatim("DTGeometryTest") << " I have " << pDD->detUnits().size() << " detUnits";
  LogVerbatim("DTGeometryTest") << " I have " << pDD->dets().size() << " dets";
  LogVerbatim("DTGeometryTest") << " I have " << pDD->layers().size() << " layers";
  LogVerbatim("DTGeometryTest") << " I have " << pDD->superLayers().size() << " superlayers";
  LogVerbatim("DTGeometryTest") << " I have " << pDD->chambers().size() << " chambers";

  // Helper lambda to format vectors for better readability
/*   auto formatLocalPointLambda = [](const LocalPoint& vec) -> std::string {
    std::ostringstream oss;
    oss << "(" << vec.x() << ", " << vec.y() << ", " << vec.z() << ")";
    return oss.str();
  }; */

  // Log Chamber Information
  LogVerbatim("DTGeometryTest") << "CHAMBERS " << std::string(120, '-');
  LogVerbatim("DTGeometryTest").log([&](auto& log) {
    for (const auto& chamber : pDD->chambers()) {
      const BoundPlane& chamberSurf = chamber->surface();
      const GlobalPoint& chamberGlobalPos = chamberSurf.position();

      // Chamber's local position relative to itself is (0,0,0)
      LocalPoint chamberLocalPos(0, 0, 0);

      log << "Chamber " << chamber->id()
          << " Global Position " << formatGlobalPoint(chamberGlobalPos)
          << " Local Position " << formatLocalPoint(chamberLocalPos)
          << " normVect " << formatVector(roundVecIfNear0(chamberSurf.normalVector()))
          << " bounds W/H/L: " << chamberSurf.bounds().width() << "/"
          << chamberSurf.bounds().thickness() << "/" << chamberSurf.bounds().length() << "\n";

      // Iterate over SuperLayers within this Chamber
      for (const auto& superLayer : chamber->superLayers()) {
        const BoundPlane& superSurf = superLayer->surface();
        const GlobalPoint& superGlobalPos = superSurf.position();

        // Transform SuperLayer's global position to Chamber's local coordinates
        LocalPoint superLocalPos = chamberSurf.toLocal(superGlobalPos);

        log << "  SuperLayer " << superLayer->id()
            << " Global Position " << formatGlobalPoint(superGlobalPos)
            << " Local Position " << formatLocalPoint(roundVecIfNear0(superLocalPos))
            << " normVect " << formatVector(roundVecIfNear0(superSurf.normalVector()))
            << " bounds W/H/L: " << superSurf.bounds().width() << "/"
            << superSurf.bounds().thickness() << "/" << superSurf.bounds().length() << "\n";

        // Iterate over Layers within this SuperLayer
        for (const auto& layer : superLayer->layers()) {
          const DTTopology& topo = layer->specificTopology();
          const BoundPlane& layerSurf = layer->surface();
          const GlobalPoint& layerGlobalPos = layerSurf.position();

          // Transform Layer's global position to Chamber's local coordinates
          LocalPoint layerLocalPos = chamberSurf.toLocal(layerGlobalPos);

          log << "    Layer " << layer->id()
              << " SL " << superLayer->id()
              << " Chamber " << chamber->id()
              << " Topology W/H/L: " << topo.cellWidth() << "/"
              << topo.cellHeight() << "/" << topo.cellLenght()
              << " first/last/# wire " << topo.firstChannel() << "/"
              << topo.lastChannel() << "/" << topo.channels()
              << " Position of first/last wire " << (topo.wirePosition(topo.firstChannel()))
              << "/" << (topo.wirePosition(topo.lastChannel()))
              << " Global Position " << formatGlobalPoint(layerGlobalPos)
              << " Local Position " << formatLocalPoint(roundVecIfNear0(layerLocalPos))
              << " normVect " << formatVector(roundVecIfNear0(layerSurf.normalVector()))
              << " bounds W/H/L: " << layerSurf.bounds().width() << "/"
              << layerSurf.bounds().thickness() << "/" << layerSurf.bounds().length() << "\n";
        }
      }
    }
  });
  LogVerbatim("DTGeometryTest") << "END " << std::string(120, '-');

  // XML Generation: Build the property tree
  if (!xmlWritten_) {
    ptree& root = tree_.get_child("DTGeometry");

    ptree chambers_node;
    for (const auto& chamber : pDD->chambers()) {
      const BoundPlane& chamberSurf = chamber->surface();
      const GlobalPoint& chamberGlobalPos = chamberSurf.position();
      LocalPoint chamberLocalPos(0, 0, 0);

      ptree chamber_node;
      chamber_node.put("<xmlattr>.rawId", chamber->id().rawId());
      chamber_node.put("<xmlattr>.Id", chamber->id());

      chamber_node.put("GlobalPosition", formatGlobalPoint(chamberGlobalPos));
      chamber_node.put("LocalPosition", formatLocalPoint(chamberLocalPos));
      chamber_node.put("NormalVector", formatVector(roundVecIfNear0(chamberSurf.normalVector()))); 

      // Bounds
      ptree bounds_node;
      bounds_node.put("<xmlattr>.width", chamberSurf.bounds().width());
      bounds_node.put("<xmlattr>.thickness", chamberSurf.bounds().thickness());
      bounds_node.put("<xmlattr>.length", chamberSurf.bounds().length());
      chamber_node.add_child("Bounds", bounds_node);

      // SuperLayers
      ptree superlayers_node;

      for (const auto& superLayer : chamber->superLayers()) {
        const BoundPlane& superSurf = superLayer->surface();
        const GlobalPoint& superGlobalPos = superSurf.position();
        LocalPoint superLocalPos = chamberSurf.toLocal(superGlobalPos);

        ptree superlayer_node;
        superlayer_node.put("<xmlattr>.rawId", superLayer->id().rawId());

        // Add superLayerNumber attribute next to rawId
        superlayer_node.put("<xmlattr>.superLayerNumber", superLayer->id().superLayer());

        superlayer_node.put("GlobalPosition", formatGlobalPoint(superGlobalPos));
        superlayer_node.put("LocalPosition", formatLocalPoint(roundVecIfNear0(superLocalPos)));
        superlayer_node.put("NormalVector", formatVector(roundVecIfNear0(superSurf.normalVector()))); 

        // SuperLayer Bounds
        ptree super_bounds_node;
        super_bounds_node.put("<xmlattr>.width", superSurf.bounds().width());
        super_bounds_node.put("<xmlattr>.thickness", superSurf.bounds().thickness());
        super_bounds_node.put("<xmlattr>.length", superSurf.bounds().length());
        superlayer_node.add_child("Bounds", super_bounds_node);

        // Layers
        ptree layers_node;

        for (const auto& layer : superLayer->layers()) {
          const DTTopology& topo = layer->specificTopology();
          const BoundPlane& layerSurf = layer->surface();
          const GlobalPoint& layerGlobalPos = layerSurf.position();
          LocalPoint layerLocalPos = chamberSurf.toLocal(layerGlobalPos);

          ptree layer_node;
          layer_node.put("<xmlattr>.rawId", layer->id().rawId());

          // Add layerNumber attribute next to rawId
          layer_node.put("<xmlattr>.layerNumber", layer->id().layer());

          layer_node.put("Topology.cellWidth", topo.cellWidth());
          layer_node.put("Topology.cellHeight", topo.cellHeight());
          layer_node.put("Topology.cellLength", topo.cellLenght());

          layer_node.put("Channels.first", topo.firstChannel());
          layer_node.put("Channels.last", topo.lastChannel());
          layer_node.put("Channels.total", topo.channels());

          // Compute wire positions relative to the chamber
          float firstWireX = topo.wirePosition(topo.firstChannel());
          float lastWireX = topo.wirePosition(topo.lastChannel());

          // Create LocalPoint in layer's local coordinates for first and last wire
          LocalPoint wireLocalLayerFirst(firstWireX, layerLocalPos.y(), layerLocalPos.z());
          LocalPoint wireLocalLayerLast(lastWireX, layerLocalPos.y(), layerLocalPos.z());


          // Transform to GlobalPoint
          GlobalPoint wireGlobalFirst = layerSurf.toGlobal(wireLocalLayerFirst);
          GlobalPoint wireGlobalLast = layerSurf.toGlobal(wireLocalLayerLast);

          // Transform to Chamber's local coordinates
          LocalPoint wireLocalChamberFirst = chamberSurf.toLocal(wireGlobalFirst);
          LocalPoint wireLocalChamberLast = chamberSurf.toLocal(wireGlobalLast);
          


          // Now the wires are refered to the chamber instead of the layer
          ptree wire_positions_node;
          wire_positions_node.put("FirstWire", topo.wirePosition(topo.firstChannel()));
          wire_positions_node.put("LastWire", topo.wirePosition(topo.lastChannel()));
          if (superLayer->id().superLayer() != 2){
            wire_positions_node.put("FirstWire_ref_to_chamber", wireLocalChamberFirst.x());
            wire_positions_node.put("LastWire_ref_to_chamber", wireLocalChamberLast.x());
          }
          else{ // Since SL2 is theta, we need to rotate the coordinates 90, the x is the y.

            wire_positions_node.put("FirstWire_ref_to_chamber", (-1)*wireLocalChamberFirst.y());
            wire_positions_node.put("LastWire_ref_to_chamber", (-1)*wireLocalChamberLast.y());
 
          }
          layer_node.add_child("WirePositions", wire_positions_node);

          layer_node.put("GlobalPosition", formatGlobalPoint(layerGlobalPos));
          layer_node.put("LocalPosition", formatLocalPoint(roundVecIfNear0(layerLocalPos)));
          layer_node.put("NormalVector", formatVector(roundVecIfNear0(layerSurf.normalVector()))); // Fixed missing ')'

          // Layer Bounds
          ptree layer_bounds_node;
          layer_bounds_node.put("<xmlattr>.width", layerSurf.bounds().width());
          layer_bounds_node.put("<xmlattr>.thickness", layerSurf.bounds().thickness());
          layer_bounds_node.put("<xmlattr>.length", layerSurf.bounds().length());
          layer_node.add_child("Bounds", layer_bounds_node);

          layers_node.add_child("Layer", layer_node);
        }
        superlayer_node.add_child("Layers", layers_node);
        superlayers_node.add_child("SuperLayer", superlayer_node);
      }
      chamber_node.add_child("SuperLayers", superlayers_node);
      chambers_node.add_child("Chamber", chamber_node);
    }
    root.add_child("Chambers", chambers_node);

    xmlWritten_ = true; // Mark as written
    LogVerbatim("DTGeometryTest") << "XML structure has been built using Boost Property Tree.";
  }
}

void DTGeometryTest::endJob() {
  if (xmlWritten_) {
    try {
      // Write the property tree to an XML file
      boost::property_tree::xml_writer_settings<std::string> settings('\t', 1);
      write_xml("DTGeometry.xml", tree_, std::locale(), settings);
      edm::LogVerbatim("DTGeometryTest") << "XML file DTGeometry.xml has been generated using Boost Property Tree.";
    }
    catch (const std::exception& e) {
      edm::LogError("DTGeometryTest") << "Failed to write XML file: " << e.what();
    }
  }
}

DEFINE_FWK_MODULE(DTGeometryTest);

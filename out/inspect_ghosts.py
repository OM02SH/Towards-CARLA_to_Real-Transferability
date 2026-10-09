import argparse
import collections
import csv
import math
import carla

def load_cells(path):
    cells = collections.defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            wx, wy = float(r["x_world"]), float(r["y_world"])
            if math.isnan(wx) or math.isnan(wy):
                continue
            cells[(round(wx / 2.0), round(wy / 2.0))].append(
                (wx, wy, float(r["length"]), float(r["width"]), float(r["speed"]), float(r["score"]), r["frame"]))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--radius", type=float, default=4.0)
    ap.add_argument("--max-lines", type=int, default=12)
    ap.add_argument("--all-labels", action="store_true", help="also list roads, sidewalks, terrain...")
    ap.add_argument("--no-carla", action="store_true")
    a = ap.parse_args()

    cells = load_cells(a.csv)
    ranked = sorted(cells.items(), key=lambda kv: -len(kv[1]))[:a.top]
    total = sum(len(v) for v in cells.values())
    print(f"{total} ghost observations in {len(cells)} cells; showing the {len(ranked)} busiest\n")

    world = env = actors = None
    if not a.no_carla:
        client = carla.Client(a.host, a.port)
        client.set_timeout(30.0)
        world = client.get_world()
        env = world.get_environment_objects(carla.CityObjectLabel.Any)
        actors = list(world.get_actors())
        print(f"map: {world.get_map().name}   environment objects: {len(env)}   actors: {len(actors)}\n")

    skip = {"Roads", "RoadLines", "Sidewalks", "Terrain", "Ground", "Sky", "Water"}
    label_hist = collections.Counter()
    for _, rows in ranked:
        mx = sum(r[0] for r in rows) / len(rows)
        my = sum(r[1] for r in rows) / len(rows)
        print(f"=== cell near world ({mx:.1f}, {my:.1f}): {len(rows)} ghost obs in {len({r[6] for r in rows})} frames, "
              f"mean size {sum(r[2] for r in rows)/len(rows):.1f}x{sum(r[3] for r in rows)/len(rows):.1f}, "
              f"mean speed {sum(r[4] for r in rows)/len(rows):.2f}, mean score {sum(r[5] for r in rows)/len(rows):.2f}")
        if a.no_carla:
            continue

        near = []
        for o in env:
            loc = o.bounding_box.location
            d = math.hypot(loc.x - mx, loc.y - my)
            if d <= a.radius:
                label = str(o.type).split(".")[-1]
                if a.all_labels or label not in skip:
                    near.append((d, label, o.name, o.bounding_box.extent))
        near.sort(key=lambda t: t[0])
        print(f"  environment objects within {a.radius} m: {len(near)}")
        for d, label, name, ext in near[:a.max_lines]:
            label_hist[label] += 1
            print(f"    {d:4.1f} m  {label:<14} {name[:40]:<40} extent {2*ext.x:.1f} x {2*ext.y:.1f} x {2*ext.z:.1f}")

        nearby_actors = []
        for ac in actors:
            loc = ac.get_location()
            d = math.hypot(loc.x - mx, loc.y - my)
            if d <= a.radius:
                nearby_actors.append((d, ac.type_id, ac.id))
        nearby_actors.sort()
        print(f"  actors within {a.radius} m: {len(nearby_actors)}")
        for d, tid, aid in nearby_actors[:a.max_lines]:
            print(f"    {d:4.1f} m  {tid}  (id {aid})")
        print()

    if label_hist:
        print("label histogram over all listed environment objects:", dict(label_hist))
        print("-> compare with hp.VEHICLE_LABELS: any vehicle-like label that is NOT in that list is missing from your GT.")


if __name__ == "__main__":
    main()